TODO:
1. 封装 态、哈密顿量、算法（包括pin scheme）、信息
2. 晶格格点序号到2x2矩阵索引的对应，态的便捷索引
3. 多近邻：SWAP gate
4. 寻路算法
   
Structure:
1. Lattice: 
   1. Latt: 真实晶格，构造哈密顿量和多近邻树（从原胞内每个点出发，延伸到哈密顿量最远相互作用的树）。
   2. fLatt：假晶格，正方晶格，能给出最近邻两点间的方向，构造方向字典：(i,j) -> direction
   3. Latt -> fLatt: 点到点映射，Latt上为真实最近邻和真实点，fLatt上为满足iPEPS张量结构的fake 最近邻和fake点。
   4. 输出：哈密顿量、多近邻树、最近邻方向字典。
2. 哈密顿量：
   1. （i，j）-> hij, i -> hi
   2. 遍历所有(i,j)时，返回hij + hi x I + I x hj
3. 态：
   1. Gamma: Vector存储，getindex(i) -> Gamma_i。
   2. lambda：Vector存储，getindex(i) -> lambda_i_r,lambda_i_u,lambda_i_d,lambda_i_l
   3. mapping: 字典，i -> (ir,iu,id,il)
4. SU流程
   1. 遍历哈密顿量的（i，j）
      1. 如果是NN近邻：
         1. 根据最近邻方向字典确定方向
         2. SU update
      2. 如果大于NN近邻：
         1. 根据最近邻树确定swap路径，面向态执行swap
         2. NN更新
         3. 根据刚才路径的逆路径，面向态执行anti swap