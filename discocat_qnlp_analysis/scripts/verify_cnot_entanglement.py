#!/usr/bin/env python3
"""
驗證 CNOT 門是否產生糾纏
使用多種方法檢測量子糾纏
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace, entropy
import matplotlib.pyplot as plt
from pathlib import Path

def schmidt_decomposition(statevector, qubit1, qubit2, num_qubits):
    """Schmidt 分解 - 檢測兩量子比特系統的糾纏"""
    # 將狀態向量重塑為兩個子系統的張量積形式
    # 對於 2-qubit 系統，直接進行 SVD
    if num_qubits == 2:
        # 重塑為 2x2 矩陣
        state_matrix = statevector.data.reshape(2, 2)
        U, s, Vh = np.linalg.svd(state_matrix)
        schmidt_coefficients = s
        schmidt_rank = np.sum(schmidt_coefficients > 1e-10)
        
        # Schmidt 熵
        schmidt_entropy = -np.sum(schmidt_coefficients**2 * np.log2(schmidt_coefficients**2 + 1e-12))
        
        return {
            'schmidt_coefficients': schmidt_coefficients,
            'schmidt_rank': schmidt_rank,
            'schmidt_entropy': schmidt_entropy,
            'is_entangled': schmidt_rank > 1
        }
    else:
        # 對於多量子比特系統，需要更複雜的處理
        return None

def check_entanglement_via_partial_trace(density_matrix, num_qubits):
    """通過部分跡檢測糾纏"""
    results = {}
    
    if num_qubits >= 2:
        # 對第一個量子比特進行部分跡
        # 保留第一個量子比特，對其他量子比特求跡
        reduced_dm_0 = partial_trace(density_matrix, [1] if num_qubits == 2 else list(range(1, num_qubits)))
        
        # 對第二個量子比特進行部分跡
        if num_qubits >= 2:
            reduced_dm_1 = partial_trace(density_matrix, [0] if num_qubits == 2 else [0] + list(range(2, num_qubits)))
        
        # 計算約化密度矩陣的純度
        reduced_dm_0_data = reduced_dm_0.data
        purity_0 = np.trace(reduced_dm_0_data @ reduced_dm_0_data).real
        if num_qubits >= 2:
            reduced_dm_1_data = reduced_dm_1.data
            purity_1 = np.trace(reduced_dm_1_data @ reduced_dm_1_data).real
        else:
            purity_1 = 1.0
        
        # 計算約化密度矩陣的熵
        entropy_0 = entropy(reduced_dm_0).real
        entropy_1 = entropy(reduced_dm_1).real if num_qubits >= 2 else 0.0
        
        # 如果約化密度矩陣不是純態（純度 < 1），則存在糾纏
        is_entangled = (purity_0 < 0.99) or (purity_1 < 0.99)
        
        results = {
            'reduced_dm_0_purity': purity_0,
            'reduced_dm_1_purity': purity_1,
            'reduced_dm_0_entropy': entropy_0,
            'reduced_dm_1_entropy': entropy_1,
            'is_entangled': is_entangled
        }
    
    return results

def calculate_linear_entropy(density_matrix):
    """計算線性熵（混合度）"""
    dm_data = density_matrix.data
    purity = np.trace(dm_data @ dm_data).real
    linear_entropy = 1.0 - purity
    return linear_entropy

def verify_cnot_entanglement():
    """驗證 CNOT 門是否產生糾纏"""
    
    print("=" * 80)
    print("CNOT 門糾纏檢測實驗")
    print("=" * 80)
    
    # 創建輸出目錄
    output_dir = Path(__file__).parent.parent / '20251113_densityMatrix'
    output_dir.mkdir(exist_ok=True)
    
    results = []
    
    # 實驗 1: |00⟩ + CNOT → 不產生糾纏（因為輸入是基態）
    print("\n實驗 1: |00⟩ → CNOT → ?")
    print("-" * 80)
    qc1 = QuantumCircuit(2)
    qc1.cx(0, 1)  # CNOT on |00⟩
    sv1 = Statevector.from_instruction(qc1)
    dm1 = DensityMatrix(sv1)
    
    print(f"最終狀態: {sv1}")
    print(f"狀態向量: {sv1.data}")
    
    # Schmidt 分解
    schmidt1 = schmidt_decomposition(sv1, 0, 1, 2)
    if schmidt1:
        print(f"Schmidt 係數: {schmidt1['schmidt_coefficients']}")
        print(f"Schmidt 秩: {schmidt1['schmidt_rank']}")
        print(f"Schmidt 熵: {schmidt1['schmidt_entropy']:.6f}")
        print(f"是否糾纏: {'是' if schmidt1['is_entangled'] else '否'}")
    
    # 部分跡檢測
    partial1 = check_entanglement_via_partial_trace(dm1, 2)
    print(f"約化密度矩陣純度 (qubit 0): {partial1['reduced_dm_0_purity']:.6f}")
    print(f"約化密度矩陣純度 (qubit 1): {partial1['reduced_dm_1_purity']:.6f}")
    print(f"約化密度矩陣熵 (qubit 0): {partial1['reduced_dm_0_entropy']:.6f}")
    print(f"約化密度矩陣熵 (qubit 1): {partial1['reduced_dm_1_entropy']:.6f}")
    print(f"是否糾纏 (部分跡): {'是' if partial1['is_entangled'] else '否'}")
    
    linear_entropy1 = calculate_linear_entropy(dm1)
    print(f"線性熵: {linear_entropy1:.6f}")
    
    results.append({
        'experiment': '|00⟩ + CNOT',
        'is_entangled': schmidt1['is_entangled'] if schmidt1 else False,
        'schmidt_rank': schmidt1['schmidt_rank'] if schmidt1 else 0,
        'linear_entropy': linear_entropy1
    })
    
    # 實驗 2: H|0⟩|0⟩ + CNOT → 產生糾纏（Bell 態）
    print("\n實驗 2: H|0⟩|0⟩ → CNOT → Bell 態")
    print("-" * 80)
    qc2 = QuantumCircuit(2)
    qc2.h(0)  # 在 |0⟩ 上應用 Hadamard
    qc2.cx(0, 1)  # CNOT
    sv2 = Statevector.from_instruction(qc2)
    dm2 = DensityMatrix(sv2)
    
    print(f"最終狀態: {sv2}")
    print(f"狀態向量: {sv2.data}")
    print(f"狀態表示: |Φ⁺⟩ = (|00⟩ + |11⟩)/√2")
    
    # Schmidt 分解
    schmidt2 = schmidt_decomposition(sv2, 0, 1, 2)
    if schmidt2:
        print(f"Schmidt 係數: {schmidt2['schmidt_coefficients']}")
        print(f"Schmidt 秩: {schmidt2['schmidt_rank']}")
        print(f"Schmidt 熵: {schmidt2['schmidt_entropy']:.6f}")
        print(f"是否糾纏: {'是' if schmidt2['is_entangled'] else '否'}")
    
    # 部分跡檢測
    partial2 = check_entanglement_via_partial_trace(dm2, 2)
    print(f"約化密度矩陣純度 (qubit 0): {partial2['reduced_dm_0_purity']:.6f}")
    print(f"約化密度矩陣純度 (qubit 1): {partial2['reduced_dm_1_purity']:.6f}")
    print(f"約化密度矩陣熵 (qubit 0): {partial2['reduced_dm_0_entropy']:.6f}")
    print(f"約化密度矩陣熵 (qubit 1): {partial2['reduced_dm_1_entropy']:.6f}")
    print(f"是否糾纏 (部分跡): {'是' if partial2['is_entangled'] else '否'}")
    
    linear_entropy2 = calculate_linear_entropy(dm2)
    print(f"線性熵: {linear_entropy2:.6f}")
    
    results.append({
        'experiment': 'H|0⟩|0⟩ + CNOT (Bell state)',
        'is_entangled': schmidt2['is_entangled'] if schmidt2 else False,
        'schmidt_rank': schmidt2['schmidt_rank'] if schmidt2 else 0,
        'linear_entropy': linear_entropy2
    })
    
    # 實驗 3: 我們的實際電路（H + RY + CNOT）
    print("\n實驗 3: 我們的實際電路 (H + RY + CNOT)")
    print("-" * 80)
    qc3 = QuantumCircuit(2)
    qc3.h(0)
    qc3.h(1)
    qc3.ry(0.5, 0)  # 基於詞性的旋轉
    qc3.ry(0.3, 1)
    qc3.cx(0, 1)  # CNOT 糾纏
    sv3 = Statevector.from_instruction(qc3)
    dm3 = DensityMatrix(sv3)
    
    print(f"最終狀態: {sv3}")
    
    # Schmidt 分解
    schmidt3 = schmidt_decomposition(sv3, 0, 1, 2)
    if schmidt3:
        print(f"Schmidt 係數: {schmidt3['schmidt_coefficients']}")
        print(f"Schmidt 秩: {schmidt3['schmidt_rank']}")
        print(f"Schmidt 熵: {schmidt3['schmidt_entropy']:.6f}")
        print(f"是否糾纏: {'是' if schmidt3['is_entangled'] else '否'}")
    
    # 部分跡檢測
    partial3 = check_entanglement_via_partial_trace(dm3, 2)
    print(f"約化密度矩陣純度 (qubit 0): {partial3['reduced_dm_0_purity']:.6f}")
    print(f"約化密度矩陣純度 (qubit 1): {partial3['reduced_dm_1_purity']:.6f}")
    print(f"約化密度矩陣熵 (qubit 0): {partial3['reduced_dm_0_entropy']:.6f}")
    print(f"約化密度矩陣熵 (qubit 1): {partial3['reduced_dm_1_entropy']:.6f}")
    print(f"是否糾纏 (部分跡): {'是' if partial3['is_entangled'] else '否'}")
    
    linear_entropy3 = calculate_linear_entropy(dm3)
    print(f"線性熵: {linear_entropy3:.6f}")
    
    results.append({
        'experiment': 'H + RY + CNOT (our circuit)',
        'is_entangled': schmidt3['is_entangled'] if schmidt3 else False,
        'schmidt_rank': schmidt3['schmidt_rank'] if schmidt3 else 0,
        'linear_entropy': linear_entropy3
    })
    
    # 總結
    print("\n" + "=" * 80)
    print("總結")
    print("=" * 80)
    print("\n檢測糾纏的方法:")
    print("1. Schmidt 分解: 如果 Schmidt 秩 > 1，則存在糾纏")
    print("2. 部分跡: 如果約化密度矩陣的純度 < 1，則存在糾纏")
    print("3. 線性熵: 測量整體混合度（但不等於糾纏）")
    print("\n實驗結果:")
    for r in results:
        print(f"  {r['experiment']}:")
        print(f"    - 是否糾纏: {'是' if r['is_entangled'] else '否'}")
        print(f"    - Schmidt 秩: {r['schmidt_rank']}")
        print(f"    - 線性熵: {r['linear_entropy']:.6f}")
    
    print("\n關鍵發現:")
    print("- CNOT 門本身不一定產生糾纏")
    print("- 如果輸入是基態 |00⟩，CNOT 不產生糾纏")
    print("- 如果輸入是疊加態（如 H|0⟩|0⟩），CNOT 會產生糾纏")
    print("- 在我們的電路中，H 門創建疊加態，然後 CNOT 產生糾纏")
    
    # 保存結果
    import json
    output_file = output_dir / 'cnot_entanglement_verification.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n結果已保存: {output_file}")

if __name__ == "__main__":
    verify_cnot_entanglement()

