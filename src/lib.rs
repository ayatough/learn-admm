use nalgebra::{DMatrix, DVector};

/// 収縮 (ソフトしきい値) 演算子
/// shrinkage(x, κ) = sign(x) * max(|x| - κ, 0)
fn shrinkage(v: f64, kappa: f64) -> f64 {
    if v > kappa {
        v - kappa
    } else if v < -kappa {
        v + kappa
    } else {
        0.0
    }
}

/// LASSO (最小二乗 + L1 正則化) を
/// ADMM で解く簡易実装.
///
/// min 0.5‖Ax - b‖₂² + λ‖x‖₁
///
/// # 引数
/// * `a` - m×n 行列 A
/// * `b` - m 次元ベクトル b
/// * `lambda` - L1 正則化係数 λ
/// * `rho` - ADMM 係数 ρ (≒ λ と同オーダで 1.0 付近が推奨)
/// * `iterations` - 反復回数
///
/// # 戻り値
/// * 推定された n 次元ベクトル x
pub fn lasso_admm(
    a: &DMatrix<f64>,
    b: &DVector<f64>,
    lambda: f64,
    rho: f64,
    iterations: usize,
) -> DVector<f64> {
    let (_, n) = a.shape();

    // 変数初期化
    let mut x = DVector::<f64>::zeros(n);
    let mut z = DVector::<f64>::zeros(n);
    let mut u = DVector::<f64>::zeros(n);

    // 事前計算: (AᵀA + ρI)⁻¹ と Aᵀb
    let ata = a.transpose() * a;
    let q = a.transpose() * b;
    let identity = DMatrix::<f64>::identity(n, n);
    // LU 分解で線形方程式を高速に解く
    let solver = (ata + rho * &identity).lu();

    for _ in 0..iterations {
        // x-update: (AᵀA + ρI)x = Aᵀb + ρ(z - u)
        let rhs = &q + rho * (&z - &u);
        x = solver.solve(&rhs).expect("線形方程式が解けません");

        // z-update: ソフトしきい値処理
        let x_hat = &x + &u;
        z = x_hat.map(|v| shrinkage(v, lambda / rho));

        // u-update: dual 変数
        u += &x - &z;
    }

    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::dmatrix;

    #[test]
    fn simple_lasso() {
        // y = 3x1 - 2x2 となるデータを用意 (ノイズなし)
        let a = dmatrix![
            1.0, 0.0;
            0.0, 1.0;
            1.0, 1.0
        ];
        let b = DVector::from_vec(vec![3.0, -2.0, 1.0]);

        let x = lasso_admm(&a, &b, 0.0, 1.0, 1000);

        assert!((x[0] - 3.0).abs() < 1e-3);
        assert!((x[1] + 2.0).abs() < 1e-3);
    }
}
