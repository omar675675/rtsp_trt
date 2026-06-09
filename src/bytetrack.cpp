#include "bytetrack.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

// ── Kalman filter ─────────────────────────────────────────────────────────────
// State  x = [cx, cy, w, h, vcx, vcy, vw, vh]  (8-dim)
// Measurement z = [cx, cy, w, h]               (4-dim)
// Constant-velocity model; noise scaled by bounding-box height (ByteTrack §3.2).

struct KalmanFilter {
    double x[8]  = {};  // state mean
    double P[64] = {};  // state covariance (8×8 row-major)

    static constexpr double SWP = 1.0 / 20.0;   // position noise weight
    static constexpr double SWV = 1.0 / 160.0;  // velocity noise weight

    void init(float cx, float cy, float w, float h) {
        x[0]=cx; x[1]=cy; x[2]=w; x[3]=h;
        x[4]=x[5]=x[6]=x[7] = 0.0;
        std::fill(std::begin(P), std::end(P), 0.0);
        double sp = SWP * h, sv = SWV * h;
        P[0*8+0] = (2*sp)*(2*sp); P[1*8+1] = (2*sp)*(2*sp);
        P[2*8+2] = (2*sp)*(2*sp); P[3*8+3] = (2*sp)*(2*sp);
        P[4*8+4] = (10*sv)*(10*sv); P[5*8+5] = (10*sv)*(10*sv);
        P[6*8+6] = (10*sv)*(10*sv); P[7*8+7] = (10*sv)*(10*sv);
    }

    // x = F*x,  P = F*P*F^T + Q
    // F exploits structure: pos_new = pos + vel, vel_new = vel
    void predict() {
        double hv = std::max(1.0, x[3]);
        for (int i = 0; i < 4; ++i) x[i] += x[i+4];

        // FP[i][j] = P[i][j] + P[i+4][j]  for i<4,  else P[i][j]
        double FP[64];
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 8; ++j) FP[i*8+j] = P[i*8+j] + P[(i+4)*8+j];
        for (int i = 4; i < 8; ++i)
            for (int j = 0; j < 8; ++j) FP[i*8+j] = P[i*8+j];

        // FPFT[i][j] = FP[i][j] + FP[i][j+4] for j<4, else FP[i][j]; then add Q
        double sp = SWP * hv, sv2 = SWV * hv;
        double q[8] = {sp*sp,sp*sp,sp*sp,sp*sp, sv2*sv2,sv2*sv2,sv2*sv2,sv2*sv2};
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 4; ++j) P[i*8+j] = FP[i*8+j] + FP[i*8+j+4];
            for (int j = 4; j < 8; ++j) P[i*8+j] = FP[i*8+j];
            P[i*8+i] += q[i];
        }
    }

    // x += K*(z - H*x),  P -= K*H*P
    void update(float z0, float z1, float z2, float z3) {
        double z[4] = {z0,z1,z2,z3};
        double sr   = SWP * std::max(1.0, x[3]);
        double rd   = sr * sr;

        // S = P[0:4,0:4] + diag(R)
        double S[16];
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) S[i*4+j] = P[i*8+j];
        for (int i = 0; i < 4; ++i) S[i*4+i] += rd;

        double Si[16];
        if (!inv4(S, Si)) return;

        // K (8×4) = P[:,0:4] * Si
        double K[32] = {};
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 4; ++j)
                for (int k = 0; k < 4; ++k)
                    K[i*4+j] += P[i*8+k] * Si[k*4+j];

        // x += K*(z - x[0:4])
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 4; ++j)
                x[i] += K[i*4+j] * (z[j] - x[j]);

        // P -= K * P[0:4,:]
        double KHP[64] = {};
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 8; ++j)
                for (int k = 0; k < 4; ++k)
                    KHP[i*8+j] += K[i*4+k] * P[k*8+j];
        for (int i = 0; i < 64; ++i) P[i] -= KHP[i];
    }

private:
    // 4×4 matrix inverse via Gauss-Jordan with partial pivoting.
    static bool inv4(const double* A, double* out) {
        double m[4][8];
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) m[i][j]   = A[i*4+j];
            for (int j = 0; j < 4; ++j) m[i][4+j] = (i==j ? 1.0 : 0.0);
        }
        for (int col = 0; col < 4; ++col) {
            int piv = col;
            for (int r = col+1; r < 4; ++r)
                if (std::abs(m[r][col]) > std::abs(m[piv][col])) piv = r;
            if (std::abs(m[piv][col]) < 1e-14) return false;
            for (int j = 0; j < 8; ++j) std::swap(m[col][j], m[piv][j]);
            double sc = 1.0 / m[col][col];
            for (int j = 0; j < 8; ++j) m[col][j] *= sc;
            for (int r = 0; r < 4; ++r) {
                if (r == col) continue;
                double f = m[r][col];
                for (int j = 0; j < 8; ++j) m[r][j] -= f * m[col][j];
            }
        }
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) out[i*4+j] = m[i][4+j];
        return true;
    }
};

// ── STrack ────────────────────────────────────────────────────────────────────

enum class TrackState : uint8_t { New, Tracked, Lost, Removed };

struct STrack {
    int        track_id          = 0;
    int        frame_id          = 0;
    int        age               = 0;
    int        hits              = 0;
    int        time_since_update = 0;
    TrackState state             = TrackState::New;
    float      conf              = 0.f;
    KalmanFilter kf;

    float cx()     const { return (float)kf.x[0]; }
    float cy()     const { return (float)kf.x[1]; }
    float kw()     const { return (float)kf.x[2]; }
    float kh()     const { return (float)kf.x[3]; }
    float left()   const { return cx() - kw() * 0.5f; }
    float top()    const { return cy() - kh() * 0.5f; }
    float width()  const { return kw(); }
    float height() const { return kh(); }

    void init(const Detection& d, int fid, int tid) {
        track_id = tid; frame_id = fid;
        age = hits = 1; time_since_update = 0;
        state = TrackState::New; conf = d.conf;
        kf.init(d.left + d.width  * 0.5f,
                d.top  + d.height * 0.5f,
                d.width, d.height);
    }

    void predict() {
        if (state != TrackState::Tracked)
            kf.x[4] = kf.x[5] = kf.x[6] = kf.x[7] = 0.0;
        kf.predict();
        ++age; ++time_since_update;
    }

    void update(const Detection& d, int fid) {
        frame_id = fid; ++hits; time_since_update = 0; conf = d.conf;
        kf.update(d.left + d.width  * 0.5f,
                  d.top  + d.height * 0.5f,
                  d.width, d.height);
        state = TrackState::Tracked;
    }
};

// ── IoU ───────────────────────────────────────────────────────────────────────

static float track_det_iou(const STrack& t, const Detection& d) {
    float tl=t.left(), tt=t.top(), tr=tl+t.width(), tb=tt+t.height();
    float dl=d.left,   dt=d.top,   dr=dl+d.width,   db=dt+d.height;
    float iw = std::max(0.f, std::min(tr,dr) - std::max(tl,dl));
    float ih = std::max(0.f, std::min(tb,db) - std::max(tt,dt));
    float inter = iw * ih;
    float uni   = t.width()*t.height() + d.width*d.height - inter;
    return inter / (uni + 1e-7f);
}

// ── Hungarian assignment (O(N³) potential method, padded to square) ───────────
// Returns assignment[r] = c (0-indexed), or -1 if row maps to a padding column.

static std::vector<int> hungarian(
    const std::vector<std::vector<float>>& cost, int nr, int nc)
{
    if (nr == 0 || nc == 0) return std::vector<int>(nr, -1);

    // The Jonker-Volgenant core below is only correct when rows <= cols.
    // When nr > nc it would pad with INF columns whose reduced cost stays INF,
    // so an augmenting path can never terminate on one (j1 stays -1 → p[-1]
    // out-of-bounds write → heap corruption). Solve the transposed problem
    // instead and invert the column→row mapping back to row→column.
    if (nr > nc) {
        std::vector<std::vector<float>> t(nc, std::vector<float>(nr));
        for (int r = 0; r < nr; ++r)
            for (int c = 0; c < nc; ++c) t[c][r] = cost[r][c];
        std::vector<int> col2row = hungarian(t, nc, nr);
        std::vector<int> ans(nr, -1);
        for (int c = 0; c < nc; ++c)
            if (col2row[c] >= 0) ans[col2row[c]] = c;
        return ans;
    }

    int N = std::max(nr, nc);
    const float INF = 1e18f;
    std::vector<float> u(N+1,0.f), v(N+1,0.f);
    std::vector<int>   p(N+1,0),   way(N+1,0);

    for (int i = 1; i <= nr; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<float> minv(N+1, INF);
        std::vector<bool>  used(N+1, false);
        do {
            used[j0] = true;
            int i0=p[j0], j1=-1;
            float delta = INF;
            for (int j = 1; j <= N; ++j) {
                if (used[j]) continue;
                float c = (i0<=nr && j<=nc) ? cost[i0-1][j-1] : INF;
                float cur = c - u[i0] - v[j];
                if (cur < minv[j]) { minv[j]=cur; way[j]=j0; }
                if (minv[j] < delta) { delta=minv[j]; j1=j; }
            }
            for (int j = 0; j <= N; ++j) {
                if (used[j]) { u[p[j]]+=delta; v[j]-=delta; }
                else          { minv[j]-=delta; }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do { int j1=way[j0]; p[j0]=p[j1]; j0=j1; } while (j0);
    }

    std::vector<int> ans(nr, -1);
    for (int j = 1; j <= nc; ++j)
        if (p[j] > 0 && p[j] <= nr) ans[p[j]-1] = j-1;
    return ans;
}

// ── BYTETracker::Impl ─────────────────────────────────────────────────────────

struct BYTETracker::Impl {
    BYTETracker::Config cfg;
    int frame_count = 0;
    int next_id     = 1;

    std::vector<STrack> tracked;  // state ∈ {New, Tracked}
    std::vector<STrack> lost;     // state == Lost

    explicit Impl(const BYTETracker::Config& c) : cfg(c) {}

    int alloc_id() { return next_id++; }

    // Match `tracks` against dets[det_indices] by IoU cost.
    // Matched tracks are updated in-place. Unmatched filled into u_tracks / u_dets.
    void match_iou(
        const std::vector<STrack*>& tracks,
        const std::vector<Detection>& dets,
        const std::vector<int>& det_idx,
        float thresh,
        std::vector<STrack*>& u_tracks,
        std::vector<int>& u_dets)
    {
        u_tracks.clear(); u_dets.clear();
        int nt = (int)tracks.size(), nd = (int)det_idx.size();
        if (nt == 0) { u_dets  = det_idx; return; }
        if (nd == 0) { u_tracks = tracks;  return; }

        std::vector<std::vector<float>> cost(nt, std::vector<float>(nd));
        for (int r = 0; r < nt; ++r)
            for (int c = 0; c < nd; ++c)
                cost[r][c] = 1.f - track_det_iou(*tracks[r], dets[det_idx[c]]);

        auto assign = hungarian(cost, nt, nd);
        std::vector<bool> det_used(nd, false);
        for (int r = 0; r < nt; ++r) {
            int c = assign[r];
            if (c >= 0 && cost[r][c] <= 1.f - thresh) {
                tracks[r]->update(dets[det_idx[c]], frame_count);
                det_used[c] = true;
            } else {
                u_tracks.push_back(tracks[r]);
            }
        }
        for (int c = 0; c < nd; ++c)
            if (!det_used[c]) u_dets.push_back(det_idx[c]);
    }

    std::vector<TrackedObject> update(const std::vector<Detection>& dets) {
        ++frame_count;

        // Split detections into high / low confidence pools
        std::vector<int> hi, lo;
        for (int i = 0; i < (int)dets.size(); ++i) {
            if      (dets[i].conf >= cfg.high_thresh) hi.push_back(i);
            else if (dets[i].conf >= cfg.low_thresh)  lo.push_back(i);
        }

        // Predict all tracks
        for (auto& t : tracked) t.predict();
        for (auto& t : lost)    t.predict();

        // Separate confirmed (Tracked) from unconfirmed (New)
        std::vector<STrack*> confirmed, unconfirmed;
        for (auto& t : tracked) {
            if (t.state == TrackState::Tracked) confirmed.push_back(&t);
            else                                unconfirmed.push_back(&t);
        }

        // ── Stage 1: (confirmed + lost) ↔ high-conf dets ─────────────────────
        // Including lost tracks here lets them recover directly when a high-conf
        // detection re-appears in their region.

        std::vector<STrack*> pool;
        pool.insert(pool.end(), confirmed.begin(), confirmed.end());
        for (auto& t : lost) pool.push_back(&t);

        std::vector<STrack*> u_pool;
        std::vector<int>     u_hi;
        match_iou(pool, dets, hi, cfg.match_thresh, u_pool, u_hi);

        // Unmatched confirmed tracks → candidates for stage 2
        // Unmatched lost tracks stay Lost (state unchanged by predict)
        std::vector<STrack*> u_confirmed;
        for (STrack* t : u_pool)
            if (t->state == TrackState::Tracked) u_confirmed.push_back(t);

        // ── Stage 2: unmatched confirmed ↔ low-conf dets ─────────────────────
        {
            std::vector<STrack*> still_u;
            std::vector<int>     dummy;
            match_iou(u_confirmed, dets, lo, 0.5f, still_u, dummy);
            for (STrack* t : still_u) t->state = TrackState::Lost;
        }

        // ── Stage 3: unconfirmed ↔ remaining high-conf dets ──────────────────
        {
            std::vector<STrack*> still_u;
            std::vector<int>     still_hi;
            match_iou(unconfirmed, dets, u_hi, cfg.match_thresh, still_u, still_hi);
            for (STrack* t : still_u) t->state = TrackState::Removed;
            u_hi = still_hi;
        }

        // ── Init new tracks from remaining unmatched high-conf dets ───────────
        for (int di : u_hi) {
            STrack t;
            t.init(dets[di], frame_count, alloc_id());
            tracked.push_back(t);
        }

        // ── Rebuild tracked / lost ────────────────────────────────────────────

        // Re-activated lost tracks → tracked
        for (auto& t : lost)
            if (t.state == TrackState::Tracked) tracked.push_back(t);
        lost.erase(std::remove_if(lost.begin(), lost.end(),
            [](const STrack& t){ return t.state == TrackState::Tracked; }),
            lost.end());

        // Newly lost / removed tracks → out of tracked
        for (auto& t : tracked)
            if (t.state == TrackState::Lost) lost.push_back(t);
        tracked.erase(std::remove_if(tracked.begin(), tracked.end(),
            [](const STrack& t){
                return t.state == TrackState::Lost ||
                       t.state == TrackState::Removed;
            }), tracked.end());

        // Expire old lost tracks
        lost.erase(std::remove_if(lost.begin(), lost.end(),
            [this](const STrack& t){ return t.time_since_update > cfg.max_age; }),
            lost.end());

        // ── Collect output ────────────────────────────────────────────────────
        std::vector<TrackedObject> out;
        for (const auto& t : tracked) {
            if (t.hits >= cfg.min_hits)
                out.push_back({t.left(), t.top(), t.width(), t.height(),
                               t.conf, t.track_id});
        }
        return out;
    }

    void reset() {
        tracked.clear();
        lost.clear();
        frame_count = 0;
        next_id     = 1;
    }
};

// ── Public API ────────────────────────────────────────────────────────────────

BYTETracker::BYTETracker(const Config& cfg)
    : impl_(std::make_unique<Impl>(cfg)) {}

BYTETracker::~BYTETracker() = default;
BYTETracker::BYTETracker(BYTETracker&&) noexcept = default;
BYTETracker& BYTETracker::operator=(BYTETracker&&) noexcept = default;

std::vector<TrackedObject> BYTETracker::update(const std::vector<Detection>& dets) {
    return impl_->update(dets);
}

void BYTETracker::reset() { impl_->reset(); }
