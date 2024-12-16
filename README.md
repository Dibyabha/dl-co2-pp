This work has been submitted to NeurIPS 2024 conference with title "Improving Power Plant CO2 Emission Estimation with Deep Learning and Satellite/Simulated Data" 

Abstract: CO2 emissions from power plants, as significant super emitters, contribute substantially to global warming. Accurate quantification of these emissions is crucial for effective climate mitigation strategies. While satellite-based plume inversion offers a promising approach, challenges arise from data limitations and the complexity of atmospheric conditions. This study addresses these challenges by (a) expanding the available dataset through the integration of NO2 data from Sentinel-5P, generating continuous XCO2 maps, and incorporating real satellite observations from OCO-2/3 for over 71 power plants in data-scarce regions; and (b) employing a customized U-Net model capable of handling diverse spatio-temporal resolutions for emission rate estimation. Our results demonstrate significant improvements in emission rate accuracy compared to previous methods (Ref). By leveraging this enhanced approach, we can enable near real-time, precise quantification of major CO2 emission sources, supporting environmental protection initiatives and informing regulatory frameworks. 

Overview of the methodology:

![overview](https://github.com/user-attachments/assets/05ba61fd-a71c-46ee-a264-3f417107f83c)

Steps to follow to replicate this work:
1. Simulated -> run org_model_train_eval.py
2. Simulated -> run shuf_model_train_eval.py
3. Simulated -> run sim_eval.py
4. Simulated -> run error_calc.py
5. Satellite -> run sat_eval.py
6. Satellite -> run error_calc.py
7. Combined -> run comb.py
8. Combined -> run model.py
9. Combined -> run error_calc.py
10. Additionally, EDA -> eda.py 
