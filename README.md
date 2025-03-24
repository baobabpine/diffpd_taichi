# Differentiable projective dynamics simulation for cloth in taichi
将布置于任意位置下落，采用projective dynamics模拟布料状态，计算最终布下落点与目标小球的距离，利用projective dynamics backwards公式反向传播计算损失函数，最终得到合适的风向使得布从原始位置下落能罩住目标小球。

## theoretical framework
![image](https://github.com/user-attachments/assets/09657cef-6204-446a-8f65-5a6ea0be3965)
![image](https://github.com/user-attachments/assets/7bb18d23-6277-4396-a7ae-938614ac2438)
![image](https://github.com/user-attachments/assets/5a98ce4c-0a66-47c2-88eb-4d983c149841)
![image](https://github.com/user-attachments/assets/183def8f-ca97-4a37-abb2-ef52333b4195)

## visualization output
### initial
![image](https://github.com/user-attachments/assets/b3e5f90b-1ad9-4251-bca9-a34caceae486)
![image](https://github.com/user-attachments/assets/1e62b34b-da3e-4df4-ab11-441ae4e47dff)
### final
![image](https://github.com/user-attachments/assets/de269505-069b-447e-8d20-82e703f79325)
