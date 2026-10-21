/* Same frozen forest threshold cascade as factory_defect.py. No CRT or IO.
 * Compile locally with scripts/build_factory_native.py; no external binaries.
 */
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
/* MSVC ABI floating-point marker; no CRT state or initialization is needed. */
int _fltused = 0;
#else
#define EXPORT __attribute__((visibility("default")))
#endif

EXPORT int factory_cascade_v1(
    const float *features, int rows, int columns,
    const int *roots, int trees,
    const int *left, const int *right, const int *feature,
    const double *threshold, const double *leaf,
    double required, double *output, long long *evaluations)
{
    long long visits = 0;
    for (int row = 0; row < rows; ++row) {
        const float *values = features + (long long)row * columns;
        double total = 0.;
        for (int tree = 0; tree < trees; ++tree) {
            int node = roots[tree];
            while (left[node] >= 0) {
                node = (double)values[feature[node]] <= threshold[node] ? left[node] : right[node];
            }
            total += leaf[node];
            ++visits;
            if (total + (trees - tree - 1) + 1e-12 < required) break;
        }
        output[row] = total / trees;
    }
    *evaluations = visits;
    return 0;
}
