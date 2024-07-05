

# Trajectory-as-a-sequence framework: Travel mode identification on trajectory data

This repository is the source code for the paper "Travel Mode Identification for Non-Uniform Passive Mobile Phone Data". The CE-RWCRF model is improved on the basis of CE-RCRF in "Trajectory-as-a-sequence: a novel travel mode identification framework".



## Paper Informatin

**Title: Travel Mode Identification for Non-Uniform Passive Mobile Phone Data**

- **Authors**: Jiaqi Zeng , Yulang Huang , Guozheng Zhang, Zhengyi Cai , Dianhai Wang*
- **Published in**: IEEE Transactions on Intelligent Transportation Systems
- **Paper link**: [https://ieeexplore.ieee.org/document/10474575/]
- **Citation**: Zeng Jiaqi, Huang Yulang, Zhang Guozheng, Cai Zhengyi, Wang Dianhai. Travel Mode Identification for Non-Uniform Passive Mobile Phone Data[J]. IEEE Transactions on Intelligent Transportation Systems, 2024: 1-14. https://doi.org/10.1109/TITS.2024.3372635

**Title: Trajectory-as-a-sequence: a novel travel mode identification framework**

- **Authors**: Jiaqi Zeng, Yi Yu, Yong Chen, Di Yang, Lei Zhang, Dianhai Wang*
- **Published in**: Transportation Research Part C: Emerging Technologies
- **Paper link**: [https://linkinghub.elsevier.com/retrieve/pii/S0968090X22003709]
- **Citation**: Zeng Jiaqi, Yu Yi, Chen Yong, Di Yang, Lei Zhang, Dianhai Wang. Trajectory-as-a-sequence: a novel travel mode identification framework[J]. Transportation Research Part C: Emerging Technologies, 2023, 146: 103957. https://doi.org/10.1016/j.trc.2022.103957



## Folder Structure

```
├── TaaSN/
│   ├── data/
│   │   ├── Geolife-corrected/
│   │   │   ├── 110/
│   │   │   ├── ...
│   │   ├── GIS/
│   ├── tassn/
│   │   ├── __init__.py
│   │   ├── evaluation/
│   │   ├── model/
│   │   ├── preprocessing/
│   │   ├── training/
│   │   ├── utils/
│   ├── log/
│   ├── 1-trips generation.py
│   ├── 2-data prepare.py
│   ├── 3-train and test.py
```



## Code Structure

- The `data` folder stores the required data. 
  - `Geolife-corrected` dataset can be found at https://github.com/RadetzkyLi/3P-MSPointNet. 
  - `GIS` includes bus stops, road networks, and rail networks. The network data is discretized at 10m intervals. Due to GitHub's upload capacity limitations, we have trimmed the data, keeping only the GIS data points closest to each trajectory point. This greatly reduces storage while hopefully not affecting calculations.

- The `taasn` folder includes the required modules, including preprocessing, model structure, trainers, evaluation, and various utilities.

- The `log` folder records the training process. 
  - It contains two existing log files, which are the results of running on our host machine, provided for readers' reference.
- There are three Python files to be run in the following order: 
  - `1-trips generation.py` reads the data and segments it into continuous trajectories.
  - `2-data prepare.py` calculates feature sequences, including motion features and GIS features.
  - `3-train and test.py` is for model training and testing, with the process being recorded in the log files.



## Contact Us

If you have questions about the code, please contact: zengjiaqi@zju.edu.cn

