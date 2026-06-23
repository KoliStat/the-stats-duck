#include "linalg.hpp"

// Eigen is confined to THIS translation unit. The public header (linalg.hpp) is
// Eigen-free and DuckDB-free so it can be shared, via git submodule, with
// downstream extensions without pulling Eigen into their build. Eigen 3.4 is
// header-only and has its own QR / Cholesky / SVD (no BLAS/LAPACK/threads), so
// this compiles to wasm_eh. See notes / the kernel roadmap (issue #16).
#include <Eigen/Dense>

namespace statsduck {
namespace linalg {

namespace {

using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// Row-major Mat -> (column-major) Eigen matrix. Copies; the values are identical,
// only the storage order differs.
MatrixXd to_eigen(const Mat &A) {
	if (A.rows == 0 || A.cols == 0) {
		return MatrixXd(static_cast<Index>(A.rows), static_cast<Index>(A.cols));
	}
	using RowMajorMap =
	    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
	return RowMajorMap(A.data.data(), static_cast<Index>(A.rows), static_cast<Index>(A.cols));
}

VectorXd to_eigen(const std::vector<double> &v) {
	if (v.empty()) {
		return VectorXd(0);
	}
	return Eigen::Map<const VectorXd>(v.data(), static_cast<Index>(v.size()));
}

std::vector<double> to_std(const VectorXd &v) {
	return std::vector<double>(v.data(), v.data() + v.size());
}

Mat to_std(const MatrixXd &M) {
	Mat out(static_cast<std::size_t>(M.rows()), static_cast<std::size_t>(M.cols()));
	for (Index i = 0; i < M.rows(); ++i) {
		for (Index j = 0; j < M.cols(); ++j) {
			out(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) = M(i, j);
		}
	}
	return out;
}

} // namespace

Solution cholesky_solve(const Mat &A, const std::vector<double> &b) {
	Solution s;
	if (A.rows == 0 || A.rows != A.cols || b.size() != A.rows) {
		return s; // ok = false
	}
	Eigen::LLT<MatrixXd> llt(to_eigen(A));
	if (llt.info() != Eigen::Success) {
		return s; // not SPD -> caller falls back to svd_solve
	}
	s.x = to_std(llt.solve(to_eigen(b)).eval());
	s.rank = static_cast<int>(A.cols);
	s.ok = true;
	return s;
}

Solution qr_solve(const Mat &A, const std::vector<double> &b) {
	Solution s;
	if (A.rows == 0 || A.cols == 0 || b.size() != A.rows) {
		return s;
	}
	Eigen::HouseholderQR<MatrixXd> qr(to_eigen(A));
	s.x = to_std(qr.solve(to_eigen(b)).eval());
	s.rank = static_cast<int>(A.cols); // QR assumes full column rank
	s.ok = true;
	return s;
}

Solution svd_solve(const Mat &A, const std::vector<double> &b, double tol) {
	Solution s;
	if (A.rows == 0 || A.cols == 0 || b.size() != A.rows) {
		return s;
	}
	Eigen::JacobiSVD<MatrixXd> svd(to_eigen(A), Eigen::ComputeThinU | Eigen::ComputeThinV);
	svd.setThreshold(tol);
	s.x = to_std(svd.solve(to_eigen(b)).eval());
	s.rank = static_cast<int>(svd.rank());
	s.ok = true;
	return s;
}

int rank(const Mat &A, double tol) {
	if (A.rows == 0 || A.cols == 0) {
		return 0;
	}
	Eigen::JacobiSVD<MatrixXd> svd(to_eigen(A)); // singular values only
	svd.setThreshold(tol);
	return static_cast<int>(svd.rank());
}

Pinv pinv(const Mat &A, double tol) {
	Pinv out;
	if (A.rows == 0 || A.cols == 0) {
		out.value = Mat(A.cols, A.rows); // n x m, all zero
		return out;
	}
	Eigen::JacobiSVD<MatrixXd> svd(to_eigen(A), Eigen::ComputeThinU | Eigen::ComputeThinV);
	const VectorXd &sv = svd.singularValues();
	const double thresh = tol * (sv.size() > 0 ? sv(0) : 0.0);
	VectorXd inv_sv(sv.size());
	for (Index i = 0; i < sv.size(); ++i) {
		if (sv(i) > thresh) {
			inv_sv(i) = 1.0 / sv(i);
			++out.rank;
		} else {
			inv_sv(i) = 0.0;
		}
	}
	// A^+ = V * S^+ * U^T  (n x m).
	MatrixXd p = svd.matrixV() * inv_sv.asDiagonal() * svd.matrixU().transpose();
	out.value = to_std(p);
	return out;
}

Mat inv_spd(const Mat &A, bool *ok) {
	if (A.rows == 0 || A.rows != A.cols) {
		if (ok) {
			*ok = false;
		}
		return Mat();
	}
	Eigen::LLT<MatrixXd> llt(to_eigen(A));
	if (llt.info() != Eigen::Success) {
		if (ok) {
			*ok = false;
		}
		return Mat();
	}
	MatrixXd inv = llt.solve(MatrixXd::Identity(static_cast<Index>(A.rows), static_cast<Index>(A.cols)));
	if (ok) {
		*ok = true;
	}
	return to_std(inv);
}

Mat sandwich(const Mat &L, const Mat &A) {
	// L: k x n, A: n x n  ->  k x k.
	if (A.rows != A.cols || L.cols != A.rows) {
		return Mat(); // dimension mismatch
	}
	if (L.rows == 0 || A.rows == 0) {
		return Mat(L.rows, L.rows);
	}
	MatrixXd Lm = to_eigen(L);
	MatrixXd Am = to_eigen(A);
	MatrixXd Sm = Lm * Am * Lm.transpose();
	return to_std(Sm);
}

std::vector<double> matvec(const Mat &A, const std::vector<double> &x) {
	if (A.cols != x.size() || A.rows == 0) {
		return std::vector<double>();
	}
	VectorXd y = to_eigen(A) * to_eigen(x);
	return to_std(y);
}

} // namespace linalg
} // namespace statsduck
