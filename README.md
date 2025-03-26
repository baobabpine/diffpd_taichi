# Differentiable projective dynamics simulation for cloth in taichi
在高性能模拟环境taichi中，采用projective dynamics simulation模拟布料状态，计算最终布下落点与目标小球的距离；第二阶段深度学习反向传播算法利用projective dynamics backwards公式训练每一次布下落的风向，计算损失函数，最终得到合适的风向使得布从原始位置下落能罩住目标小球。

## Theoretical framework
![image](https://github.com/user-attachments/assets/4b3de985-4a43-405f-a912-bab2499f5e78)
![image](https://github.com/user-attachments/assets/cf5df41c-d0c2-4d4f-a1ab-58d2989d3a6e)

![image](https://github.com/user-attachments/assets/3370e642-a0f5-484b-b283-37f7dcf269f9)
![image](https://github.com/user-attachments/assets/60a0804c-b238-4f8f-b660-85e7ec8f12cd)

## Implementation
![image](https://github.com/user-attachments/assets/93d6be52-55ff-4dd2-a12e-4ea5651d29c7)

### reference
Du, Tao, et al. "Diffpd: Differentiable projective dynamics." ACM Transactions on Graphics (ToG) 41.2 (2021): 1-21.
Li, Yifei, et al. "Diffcloth: Differentiable cloth simulation with dry frictional contact." ACM Transactions on Graphics (TOG) 42.1 (2022): 1-20.

## visualization output
### initial
![image](https://github.com/user-attachments/assets/b3e5f90b-1ad9-4251-bca9-a34caceae486)
![image](https://github.com/user-attachments/assets/1e62b34b-da3e-4df4-ab11-441ae4e47dff)
### final
![image](https://github.com/user-attachments/assets/de269505-069b-447e-8d20-82e703f79325)
