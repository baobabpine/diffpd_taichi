# diffprojectdynamic_taichi
将布置于任意位置下落，采用projective dynamics模拟布料状态，计算最终布下落点与目标小球的距离，利用projective dynamics backwards公式反向传播计算损失函数，最终得到合适的风向使得布从原始位置下落能罩住目标小球。

##theoretical framework
![image](https://github.com/user-attachments/assets/b57fb756-493e-46ca-a339-b02080cfb1da)
![image](https://github.com/user-attachments/assets/66294faa-a8ed-419e-9c60-f24168a6a0e2)
![image](https://github.com/user-attachments/assets/b73d9303-2cd5-49a8-aa1f-b39d505fd5e0)
![image](https://github.com/user-attachments/assets/76c86f62-388f-4088-8084-5f778aa68714)

##visualization output
initial
![image](https://github.com/user-attachments/assets/b3e5f90b-1ad9-4251-bca9-a34caceae486)
![image](https://github.com/user-attachments/assets/1e62b34b-da3e-4df4-ab11-441ae4e47dff)
final
![image](https://github.com/user-attachments/assets/de269505-069b-447e-8d20-82e703f79325)
