import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from os.path import dirname, join as pjoin
from scipy.io import loadmat
import numpy as np
from scipy.optimize import curve_fit, root
from scipy.interpolate import interp1d
from scipy.stats import poisson
from pnr_spdc_model_functions import *
import csv

#Params
rate = 10**6
#


# eta_i_data = 0.3300177738
# eta_i_data_err = 0.0009380973259
# eta_s1_data = 2*0.1793980285
# eta_s1_data_err = 2*0.0003875799469
# eta_s2_data = 2*0.2242867015
# eta_s2_data_err = 2*0.0004135296631
# # eta_i_data = 0.32#0.3300177738#
# # eta_s1_data = 2*0.182 #2*0.1793980285#
# # eta_s2_data = 2*0.215 #2*0.2242867015#

#Data
shg_currents = np.array([0.68,0.6825,0.69,0.6925,	0.705,	0.72,	0.735, 0.7,	0.72,	0.74,0.76,	0.77,	0.78,	0.795,	0.8,	0.81,	0.82,	0.8275,	0.84,	0.85,	0.855,	0.865,	0.8725,	0.875	,0.89,	0.9,	0.92,	0.965,	0.995,	1.07,	1.09,	1.095,	1.115,	1.145,	1.175,	1.145,	1.16])

#Idler Singles
Ci_multiphoton_data_arr = np.array([1.145449489	,2.74860543,1.762473761,2.028990509,3.651366576,	13.92144767,	19.55893406,28.8,	224.7,	1038,6428,	11341	,23329.66667,	53046,	82356	,103673	,123035,	181691,	237824,	289754,	333645	,385460,	426635	,517637,	596021,	702930,	801882,	902625,	962408,	997930	,996279	,999826	,1008687,	1005769,	1014329	,998465.5,	1008010	])
Ci_multiphoton_err_arr = np.array([0.008317331902,0.03197133705,0.02595628058,0.01956600107,	0.04357506823,	0.06343137857,	0.1387476823,1.697056275,	4.740253158,	32.21800739	,80.17480901,106.4941313,	88.18478075	,132.9736816,	286.977351,	185.8969248	,350.7634531	,426.2522727	,487.6720209,	538.288027,	577.6201174	,620.8542502,	653.1730246	,719.4699438	,772.0239634,	838.409208,	895.478643,	950.0657872,	981.0239549,	998.9644638,	998.137766,	999.9129962	,1004.334108,	1002.880352	,1007.139017,	315.9850471,	1003.997012])
Ci_singlephoton_data_arr = np.array([385.789717	,1048.109334,1359.805264,1712.887144,1994.838932	,4610.649805,	6049.783054,3877.2	,16578.9,	40823,107666,	144201,	197018.6667	,272722.6667,	312391	,336416.3333,	352768,	381195	,393407,	392027,	386281,	378230,	361624,	330924,	297434,	227944,	160762,	82808,	43147,	4894,	3147,	2798,	1830,	870,	347	,146.4,	20.86666667])
Ci_singlephoton_err_arr = np.array([0.152641064,0.624320928,0.7209738105,0.5684947261,	1.018507604,	1.154364651	,2.440186649,19.6906069	,40.71719538	,202.0470242,328.1249762,	379.7380676	,256.2672216	,301.5087985	,558.9194933	,334.8712854,	593.9427582,	617.4099125,	627.2216514,	626.1205954	,621.5150843	,615.004065,601.3518105,	575.2599412,	545.3751003,	477.4348123	,400.9513686	,287.763792,	207.7185596,	69.95712973,	56.09812831,	52.89612462	,42.77849927,	29.49576241	,18.62793601,	3.826225294	,0.8339997335])
Ci_total_data_arr = np.array([387.0629215,1051.077352,1361.63265,1715.028473,	1998.674525	,4625.190212	,6069.803553,3906,	16803.6,	41861,114094,	155542,	220348.3333	,325768.6667,	394747,	440087.3333	,475803	,562886,	631231,	681781	,719926	,763690,	788259,	848561,	883455	,930874	,962644	,985433	,1005555	,1002824	,999426	,1002624,	1010517	,1006639	,1014676	,998611.9,	1008032])
Ci_total_err_arr = np.array([0.1528927338,0.6252042737,0.7214580913,0.5688499608,	1.019486306,	1.156183453,	2.444220959, 19.76360291,	40.99219438,	204.599609,337.7780336,	394.3881337	,271.0155797	,329.5292939,	628.2889463	,383.0088482	,689.784749,	750.2572892,	794.500472,	825.7003088	,848.4845314	,873.8935862	,887.8395125,	921.1737078,	939.9228692,	964.8181176,	981.14423	,992.6897803	,1002.773653	,1001.411005,	999.7129588	,1001.31114	,1005.244746	,1003.314009	,1007.311273	,316.0082119	,1004.007968])

#mu_exact_arr = np.array([0.007428040854,	0.01355337206,	0.02542684271,	0.05970315606,0.07864716611,	0.1184134837,	0.194505285,	0.2636311545,	0.3081628022,0.3487702966	,0.4766353179,	0.6045240679	,0.7391174587,	0.8637365027	,1.019115353,1.179775126	,1.564217162,	1.970255586	,3.083783736	,4.988007116	,10.90021496	,22.3053283,	203.908868	,316.5805529,	357.3359543	,551.195082,	1156.056322,	2923.138329,	6820.119536,	48307.24281])

#Signal Singles
Cs1_total_data_arr = np.array([202.4614084,596.6128672,776.4743884,962.4305436,1147.316712	,2646.592021,	3473.096344, 2315.4,	9907.8,	24766,65954,	89811,	137283.3333,	207711.6667,	254614,	290536,	316653,	369555,	427077,	473014,	508853,	573134,	582008,	647976,	699773,	772649,	838170,	907191,	955480,	990686,	992298,	996250,	1005803,	1003699	,1012181,	998530,	1008552])
Cs1_total_err_arr = np.array([0.1105776071,0.4710325362,	0.5448095325,0.4261345663,0.7724173453,	0.8745918985,	1.848892077, 15.21643848,	31.47665802,	157.3721703,256.8151086,	299.6848345,	213.9184684,	263.1296681,	504.5929052,	311.1998286	,562.7192906,	607.9103552	,653.5112853,	687.7601326,	713.3393302,	757.0561406,	762.8944881,	804.9695646,	836.5243571,879.0045506	,915.5162478,	952.4657474,	977.4865728	,995.3321054,	996.1415562,	998.1232389,	1002.897303	,1001.847793,	1006.072065	,315.9952531,	1004.266897])
Cs2_total_data_arr = np.array([245.1676285,728.9033098,838.3390925,1135.07541,	1356.472356 ,	3083.942628 ,	4033.983259, 2650.1,	11517.3	,28844,76542,	103575,	155860.3333,	234471.3333,	285949,	324201.3333,	354215,	415256,	474802,	521774	,557840,	619257,	632452	,697529	,746701	,817419	,873340,	930303,	971453,	995789,	994667,	996331,	1005335	,1004109,	1009068,	996666.3,	1006310])
Cs2_total_err_arr = np.array([0.1216823599,0.5206424321,0.5660971771,0.4627798307,	0.8398772771,	0.9440936915,	1.992600302, 16.27912774,	33.93714779,	169.8352143,	276.662249,321.8307008,	227.9329824,	279.5659334,	534.7419939	,328.7356249,595.1596424,	644.4036002,	689.0587783,	722.3392555	,746.8868723,	786.9288405,	795.2685081,	835.1820161,	864.1186261,	904.1122718,	934.5266181,	964.5221615,985.6231531,	997.8922788,	997.3299354,	998.1638142,	1002.663952,	1002.052394,	1004.523768,	315.7002217,	1003.150039])

#Signal-Idler twofolds
Cis1_binary_data_arr = np.array([64.69477499,186.3594279,248.8767123,315.0958852,	356.4669787,	871.2312807,	1115.211406, 714.3,	3230,	8700,25250,	38750,	60500,	108000,	147500,	173333.3333,	197500,	245000,	306000,	361500,	402500,	465000,	487500,	570000,	637500,	725000	,815000,	900000,	945000,	990000	,985000	,995000,	995000,	996000,	997000,	998000,	997500])
Cis1_binary_err_arr = np.array([0.06251295244,0.2648377607,0.2637744101,0.2438739367,	0.4305464478,	0.5018702462,	1.04717225, 8.451627062,	17.97220076,	93.27379053	,158.9024858,196.8501969,	245.9674775,	328.6335345,	384.0572874	,240.370085,	444.4097209,	494.9747468	,553.1726674,	601.2487006	,634.428877	,681.9090848,	698.2120022,	754.9834435	,798.4359711,	851.4693183	,902.7735043,	948.6832981	,972.1111048,	994.9874371,	992.4716621,	997.4968672	,997.4968672,	997.997996	,998.4988733	,998.9994995	,998.7492178])
Cis1_pnr_data_arr = np.array([64.57390516,185.9435454,248.5412357,314.5432239,	355.7628705,	867.3073143,	1109.39823, 710.8	,3179,	8400,23000,	34200,	51500,	83500,	108000,	121500,	134000,	152500,	172500,	188000,	195000,	210000,	202500,	199000,	186000,	161500,	124000,	71000,	38500,	4800,	3250,	2800,	1850,	875	,350,	155	,27.93333333])
Cis1_pnr_err_arr = np.array([0.06245452839,0.2645420876,0.2635965711,0.2436599719,0.4301210211,	0.5007387761,	1.044439427, 8.430895563,	17.82975042	,91.6515139	,151.6575089,184.9324201,	226.9361144,	288.9636655,	328.6335345,	201.246118	,366.0601044,	390.5124838,	415.3311931,	433.5896678,	441.5880433,	458.2575695	,450	,446.0941605,	431.2771731	,401.8706259,	352.1363372,	266.4582519,	196.2141687,	69.2820323	,57.00877125,	52.91502622	,43.01162634,	29.58039892,	18.70828693,	5.567764363,	0.9649409884	])

Cis2_binary_data_arr = np.array([92.48648206,227.6741071,297.524283,368.8360324,	444.2258627,	997.5067712	,1337.019885, 838.2,	3708.3,	10000,26250,	44500,	69000,	119000,	167000,	192666.6667,	215000,	272000,	335000,	390000,	442500,	500000	,530000,	610000,	680000,	765000,	850000,	920000,	960000,	992000,	998000,	998000,	997500,	1000000,	1000000	,999000,	1000000])
Cis2_binary_err_arr = np.array([0.08704672726,0.2910329684,0.3373070062,0.8640784944,	0.393492669	,0.5370103365,	0.8364473116, 9.155326319	,19.2569468	,100,162.0185175,	210.9502311,	262.6785107,	344.9637662,	408.6563348,	253.4210374	,463.6809248,	521.5361924	,578.7918451,	624.4997998,	665.2067348,	707.1067812,	728.0109889,	781.0249676	,824.6211251,	874.6427842,	921.9544457	,959.1663047,	979.7958971,	995.9919678	,998.9994995,	998.9994995	,998.7492178,	1000,	1000,	999.4998749,	1000])
Cis2_pnr_data_arr = np.array([92.0693921,227.1019345,297.1154876,368.2044534,	443.2063437, 	993.1275837,	1329.99686, 834.8,	3652.5,	9700,28500,	39500,	58750,	94000,	122000,	137000,	149000,	170000,	192500,	208000,	214500,	230000,	222000,	215500,	201000,	172500,	130000,	73500,	39000,	4900,	3300,	2800,	1950,	850,	360,	159,	27])
Cis2_pnr_err_arr = np.array([0.08685022643,0.2906670377,0.3370751983,0.863338372,	0.3930408676,	0.8413441642,	0.8342475969,9.136739024,	19.11151485	,98.48857802,168.8194302,	198.7460691	,242.3839929,	306.5941943	,349.2849839,	213.6976057,	386.0051813,	412.3105626	,438.7482194,	456.07017,	463.1414471,	479.5831523	,471.1687596	,464.2197755,	448.3302354,	415.3311931	,360.5551275,	271.1088342	,197.4841766,	70,	57.44562647,	52.91502622,	44.15880433,	29.15475947,	18.97366596	,5.639148872,	0.9486832981])
#Signal1-Signal1 twofolds
Cs1s2_data_arr = np.array([0.9351900143,2.822172619,3.788921189,4.947859116,	6.16360601,	18.73467709,	27.96687158, 14.7,	156.3636364	,860,	4400,9600,	21050,	48250,	76000,	95700,	113500,	144000,	195000,	246000,	285000,	356500,	367500,	450000,	530000,	625000,	730000,	845000,	915000,	990000,	985000,	985000,	996000,	995000,	996000,	1001000,	1005000])
Cs1s2_err_arr = np.array([0.007515975191,0.0324023982, 0.03806465863,0.03055995325,	0.05661451522, 	0.07359491787, 	0.1658293764, 1.212435565,	3.770262064,	29.3257566,66.33249581,	97.97958971,	145.0861813,	219.6588264,	275.680975,	97.82637681	,106.5363788,	379.4733192,	441.5880433,	495.9838707,	533.8539126	,597.0762095,	606.2177826,	670.8203932,	728.0109889	,790.569415,	854.4003745,	919.2388155,	956.5563235	,994.9874371,	992.4716621,	992.4716621,	997.997996	,997.4968672,	997.997996,	1000.499875,	1002.496883])
#threefolds
Cis1s2_binary_data_arr = np.array([0.03463022843,0.2273065476,0.3909883721,0.6001381215,	0.8645659432,	4.453311394,	7.74965847, 3.066666667,	60.9,	410,	2500,5750,	12350,	31250,	52500,	67330,	82000,	110000,	156000	,202500	,240000,	299000	,322000	,405000,	490000,	595000,	715000,	836000	,910000	,985000	,984000,	980000,	995000,	994000,	995000,	995000,	1000000])
Cis1s2_binary_err_arr=np.array([0.00144631565,0.009195838612,0.01222773658,0.01064313646,	0.02120359107,	0.0358811415,	0.08729327861, 0.3197221016,	1.424780685,	9.055385138	,50,75.82875444,	111.1305539,	176.7766953,	229.1287847,	82.0548597,	286.3564213	,331.662479,	394.9683532	,450,	489.8979486	,546.8089246,	567.4504384	,636.3961031,	700	,771.362431,	845.5767263,	914.3303561,	953.9392014	,992.4716621,	991.9677414,	989.9494937,	997.4968672,	996.9954864	,997.4968672,	997.4968672	,1000])
Cis1s2_pnr_data_arr = np.array([0.02847606133,0.1863839286,0.3260658915,0.492058011,	0.7249582638,	3.702385849,	6.453210383, 2.433333333,	49.75,	344,1900,	4250,	9250,	21500,	33500,	42000,	49500,	60000,	78000,	93000,	103000	,124000	,119000	,126000,	129000,	120000,	100000,	61500,	35000,	4800,	3250,	2700,	1830,	790,	345	,163.2,	28.26666667])
Cis1s2_pnr_err_arr = np.array([0.001311521358,0.008327019824,0.01116649405,0.009637228947,	0.01941631728,	0.03271640364,	0.07965763999, 0.2848001248,	1.287762918,	8.294576541	,43.58898944,65.19202405,	96.17692031	,146.628783	,183.0300522,	64.80740698,	222.4859546,244.9489743,	279.2848009,	304.9590136,	320.9361307,	352.1363372,	344.9637662,354.964787,	359.1656999	,346.4101615,	316.227766,	247.9919354,	187.0828693,	69.2820323,	57.00877125,	51.96152423	,42.77849927,	28.10693865	,18.57417562,	5.713142743,	0.9706813186	])
#g2

eta_i_arr = 0.5*(Cis1_binary_data_arr/Cs1_total_data_arr + Cis2_binary_data_arr/Cs2_total_data_arr )
eta_i_arr_single =0.5*(Cis1_pnr_data_arr/Cs1_total_data_arr + Cis2_pnr_data_arr/Cs2_total_data_arr )

eta_s1_arr = Cis1_binary_data_arr/Ci_total_data_arr
eta_s1_arr_single = Cis1_pnr_data_arr/Ci_singlephoton_data_arr

eta_s2_arr = Cis2_binary_data_arr/Ci_total_data_arr
eta_s2_arr_single = Cis2_pnr_data_arr/Ci_singlephoton_data_arr


print("eta_i_arr: ",eta_i_arr )
print("eta_i_arr_single: ",eta_i_arr_single )

print("eta_s1_arr: ",eta_s1_arr )
print("eta_s1_arr_single: ",eta_s1_arr_single )

print("eta_s2_arr: ",eta_s2_arr )
print("eta_s2_arr_single: ",eta_s2_arr_single )


eta_i_data = np.mean(eta_i_arr[:7])#0.3300177738
eta_i_data_err=np.std(eta_i_arr[:7])#0.0009380973259
eta_s1_data = 2*np.mean(eta_s1_arr[:7])#2*0.1793980285
eta_s1_data_err = 2*np.std(eta_s1_arr[:7])#2*0.0003875799469
eta_s2_data = 2*np.mean(eta_s2_arr[:7])#2*0.2242867015
eta_s2_data_err =2*np.std(eta_s2_arr[:7]) #2*0.0004135296631



fig, ax = plt.subplots(1,1)
ax.plot(eta_i_arr)
ax.plot(eta_s1_arr)
ax.plot(eta_s2_arr)



print(eta_i_data,eta_s1_data/2, eta_s2_data/2)

# eta_i_data = 0.3300177738
# eta_i_data_err=0.0009380973259
# eta_s1_data = 2*0.1793980285
# eta_s1_data_err = 2*0.0003875799469
# eta_s2_data = 2*0.2242867015
# eta_s2_data_err = 2*0.0004135296631


eta_i_fit = 0.322
eta_s1_fit = 2*0.181
eta_s2_fit = 2*0.22

# eta_i_data = np.mean([np.append(eta_i_arr[0:2],eta_i_arr[4:6]) ])
# eta_i_data_err = np.std(eta_i_arr[:7])
# eta_s1_data = np.mean(eta_s1_arr[:7])
# eta_s1_data_err = np.std(eta_s1_arr[:7])
# eta_s2_data = np.mean(eta_s2_arr[:7])
# eta_s2_data_err =np.std(eta_s2_arr[:7])





g2_thermal_data_arr = Cs1s2_data_arr*rate/(Cs1_total_data_arr*Cs2_total_data_arr)
g2_thermal_err_arr = g2_thermal_data_arr*np.sqrt((Cs1s2_err_arr**2/Cs1s2_data_arr**2)+(Cs1_total_err_arr**2/Cs1_total_data_arr**2)+(Cs2_total_data_arr**2/Cs2_total_data_arr**2))
g2_binary_data_arr = Cis1s2_binary_data_arr*Ci_total_data_arr/(Cis1_binary_data_arr*Cis2_binary_data_arr)
g2_binary_err_arr= g2_binary_data_arr*np.sqrt((Cis1s2_binary_err_arr**2/Cis1s2_binary_data_arr**2)+(Ci_total_err_arr**2/Ci_total_data_arr**2)+(Cis1_binary_err_arr**2/Cis1_binary_data_arr**2)+(Cis2_binary_err_arr**2/Cis2_binary_data_arr**2))
g2_pnr_data_arr = Cis1s2_pnr_data_arr*Ci_singlephoton_data_arr/(Cis1_pnr_data_arr*Cis2_pnr_data_arr)
print(len(Cis1s2_pnr_err_arr))
print(len(Ci_singlephoton_err_arr))
print(len(Cis1_pnr_err_arr))
print(len(Cis2_pnr_err_arr))
g2_pnr_err_arr = g2_pnr_data_arr*np.sqrt((Cis1s2_pnr_err_arr**2/Cis1s2_pnr_data_arr**2)+(Ci_singlephoton_err_arr**2/Ci_singlephoton_data_arr**2)+(Cis1_pnr_err_arr**2/Cis1_pnr_data_arr**2)+(Cis2_pnr_err_arr**2/Cis2_pnr_data_arr**2))

svd = []
with open("JSI_paper_final.txt") as csvfile:
    filereader = csv.reader(csvfile, delimiter='\n')
    for row in filereader:
        svd.append(float(row[0]))
svd = np.array(svd)

single_to_total_data_arr = Ci_singlephoton_data_arr/Ci_total_data_arr
multi_to_total_data_arr = Ci_multiphoton_data_arr/Ci_total_data_arr
mu_arr = (np.ones(len(single_to_total_data_arr))- single_to_total_data_arr)/(eta_i_data*single_to_total_data_arr)


g2_binary_low_mu_data_arr = g2_binary_data_arr[:5]
g2_binary_low_mu_err_arr = g2_binary_err_arr[:5]
g2_pnr_low_mu_data_arr = g2_pnr_data_arr[:5]
g2_pnr_low_mu_err_arr = g2_pnr_err_arr[:5]


plt.plot(Cs1s2_data_arr,"o")


inds= list(range(len(Cs1s2_data_arr)-16))
# inds1 = list(range(16,21))
# inds2 = [24,26]
# inds=[*inds, *inds1,*inds2]
print(inds)
data_indices = inds
shg_currents=shg_currents[data_indices]
Ci_multiphoton_data_arr=Ci_multiphoton_data_arr[data_indices]
Ci_multiphoton_err_arr=Ci_multiphoton_err_arr[data_indices]
Ci_singlephoton_data_arr = Ci_singlephoton_data_arr[data_indices]
Ci_singlephoton_err_arr = Ci_singlephoton_err_arr[data_indices]
Ci_total_data_arr = Ci_total_data_arr[data_indices]
Ci_total_err_arr = Ci_total_err_arr[data_indices]
single_to_total_data_arr = single_to_total_data_arr[data_indices]
multi_to_total_data_arr = multi_to_total_data_arr[data_indices]
mu_arr = mu_arr[data_indices]
Cs1_total_data_arr = Cs1_total_data_arr[data_indices]
Cs1_total_err_arr = Cs1_total_err_arr[data_indices]
Cs2_total_data_arr = Cs2_total_data_arr[data_indices]
Cs2_total_err_arr = Cs2_total_err_arr[data_indices]
Cis1_binary_data_arr = Cis1_binary_data_arr[data_indices]
Cis1_binary_err_arr = Cis1_binary_err_arr[data_indices]
Cis1_pnr_data_arr = Cis1_pnr_data_arr[data_indices]
Cis1_pnr_err_arr = Cis1_pnr_err_arr[data_indices]
Cis2_binary_data_arr = Cis2_binary_data_arr[data_indices]
Cis2_binary_err_arr = Cis2_binary_err_arr[data_indices]
Cis2_pnr_data_arr = Cis2_pnr_data_arr[data_indices]
Cis2_pnr_err_arr = Cis2_pnr_err_arr[data_indices]
Cs1s2_data_arr = Cs1s2_data_arr[data_indices]
Cs1s2_err_arr = Cs1s2_err_arr[data_indices]
Cis1s2_binary_data_arr=Cis1s2_binary_data_arr[data_indices]
Cis1s2_binary_err_arr=Cis1s2_binary_err_arr[data_indices]
Cis1s2_pnr_data_arr=Cis1s2_pnr_data_arr[data_indices]
Cis1s2_pnr_err_arr=Cis1s2_pnr_err_arr[data_indices]
g2_thermal_data_arr = g2_thermal_data_arr[data_indices]
g2_thermal_err_arr = g2_thermal_err_arr[data_indices]
g2_binary_data_arr = g2_binary_data_arr[data_indices]
g2_binary_err_arr = g2_binary_err_arr[data_indices]
g2_pnr_data_arr = g2_pnr_data_arr[data_indices]
g2_pnr_err_arr = g2_pnr_err_arr[data_indices]
eta_i_arr = eta_i_arr[data_indices]
eta_i_arr_single = eta_i_arr_single[data_indices]
eta_s1_arr = eta_s1_arr[data_indices]
eta_s1_arr_single = eta_s1_arr_single[data_indices]
eta_s2_arr = eta_s2_arr[data_indices]
eta_s2_arr_single = eta_s2_arr_single[data_indices]

#plt.plot(Cs1s2_data_arr)


data_arr = [Cis1s2_binary_data_arr,Cis1s2_pnr_data_arr]
mu_guess_arr=mu_arr#[1.25390133*10**(-3), 6.26482215*10**(-3), 1.42171167*10**(-2), 1.87538035*10**(-2),1.17981844*10**(-2), 5.25544984*10**(-2), 1.36285636*10**(-1), 3.36578372*10**(-1),5.11272754*10**(-1), 7.52443805*10**(-1) ,1.21272995 ,1.59545378,1.82552176, 2.40288011, 2.95302181, 3.47388205,3.88212204, 4.51971003]

mu_solve_arr= solveSystemEqArr_analytic(data_arr, mu_guess_arr, eta_i = eta_i_data, eta_s1 = eta_s1_data, eta_s2 = eta_s2_data, lambdas=svd)
print("mu_solve_arr")
print(mu_solve_arr)
print("g2 binary")
print(g2_binary_data_arr)
print("g2 pnr")
print(g2_pnr_data_arr)
print()
# print(mu_solve_arr[9],g2_binary_data_arr[9],g2_pnr_data_arr[9] )
# print()
# print()
# print(mu_solve_arr[8],g2_binary_data_arr[8],g2_pnr_data_arr[8] )
# print()


def Ci_pnr_analytic(mu, k):
    rate = 10**6
    lambdas = svd
    term1 = 1
    term2 = 1
    for l in lambdas:
        m = l*mu
        term1 *= 2**k/(2**k + (2**k-1)*m*eta_i_data)
        term2 *= 1/(1+m*eta_i_data)
    return rate*2**k*(term1- term2)


popt_idler_singlephoton, pcov_idler_singlephoton= curve_fit(Ci_pnr_analytic,mu_solve_arr,Ci_singlephoton_data_arr,p0=10)


k_single = popt_idler_singlephoton[0]
k_single_err = np.sqrt(pcov_idler_singlephoton[0][0])




eta_i = eta_i_data
eta_i_err = eta_i_data_err
eta_s1 = eta_s1_data
eta_s1_err =eta_s1_data_err
eta_s2 = eta_s2_data
eta_s2_err =eta_s2_data_err
k = k_single
k_err =k_single_err
mu_arr = mu_solve_arr





print("k", k, k_err)
print("eta_i", eta_i, eta_i_err)
print("eta_s1", eta_s1, eta_s1_err)
print("eta_s2", eta_s2, eta_s2_err)










mu_model_arr = np.linspace(10**(-6), np.max(mu_arr) + 0.1, 10000)



idler_title= "Idler"
signals_title = "Signals"#+r", $\eta_{s1} = $"+"{:.3f}".format(eta_s1/2)+r"$\pm$"+"{:.3f}".format(eta_s1_err/2)+r", $\eta_{s2} = $"+"{:.3f}".format(eta_s2/2)+r"$\pm$"+"{:.3f}".format(eta_s2_err/2)
twofolds_is1_title="Idler & Signal 1"# +r", $\eta_{s1}$ = "+"{:.3f}".format(eta_s1/2)+r"$\pm$"+"{:.3f}".format(eta_s1_err/2)+r", $\eta_{i} = $"+"{:.3f}".format(eta_i)+r"$\pm$"+"{:.3f}".format(eta_i_err)
twofolds_is2_title = "Idler & Signal 2"# +r", $\eta_{s2} = $"+"{:.3f}".format(eta_s2/2)+r"$\pm$"+"{:.3f}".format(eta_s2_err/2)+r", $\eta_{i} = $"+"{:.3f}".format(eta_i)+r"$\pm$"+"{:.3f}".format(eta_i_err)
twofolds_s1s2_title="Signal 1 & Signal 2"# +r", $\eta_{s1}$ = "+"{:.3f}".format(eta_s2/2)+r"$\pm$"+"{:.3f}".format(eta_s2_err/2)+r", $\eta_{s2} = $"+"{:.3f}".format(eta_s2/2)+r"$\pm$"+"{:.3f}".format(eta_s2_err/2)
threefolds_title =" Idler & Signal 1 & Signal 2"# +r", $\eta_{s1}$ = "+"{:.3f}".format(eta_s1/2)+r"$\pm$"+"{:.3f}".format(eta_s1_err/2)+r", $\eta_{s2} = $"+"{:.3f}".format(eta_s2/2)+r"$\pm$"+"{:.3f}".format(eta_s2_err/2)+r", $\eta_{i} = $"+"{:.3f}".format(eta_i)+r"$\pm$"+"{:.3f}".format(eta_i_err)
g2_thermal_title = r"$g^{2}(0)$ Thermal"
title_arr = [[idler_title,signals_title], [twofolds_is1_title,twofolds_is2_title], [twofolds_s1s2_title,threefolds_title]]

fig, ax = plt.subplots(3,2, figsize = (8,7.5))
plt.suptitle('CQNET 2021', fontsize = 14, x = 0.525,y=0.99)
# plt.annotate('CQNET 2021',
#             xy=(.5, .97), xycoords='figure fraction',
#             horizontalalignment='left', verticalalignment='top',
#             fontsize=12)

ax[0][0].errorbar(mu_arr,Ci_total_data_arr/rate,yerr = Ci_total_err_arr/rate,fmt = 'go', label = r"P($N>0$)")
ax[0][0].plot(mu_model_arr, Ri_binary_analytic(mu_model_arr,eta_i,lambdas = svd), "-g")#,label=r"P($N>0$) model" )
ax[0][0].errorbar(mu_arr,Ci_singlephoton_data_arr/rate, yerr =Ci_singlephoton_err_arr/rate,fmt =  'bo', label = r"P($N=1$)")
ax[0][0].plot(mu_model_arr,Ri_pnr_analytic(mu_model_arr,eta_i,k=k, lambdas = svd), "-b")#,label = r"P($N=1$)"+" model")#", $k=$"+"{:.3f}".format(k)+r"$\pm$"+"{:.3f}".format(k_err))
ax[0][0].errorbar(mu_arr,Ci_multiphoton_data_arr/rate, yerr=Ci_multiphoton_err_arr/rate ,fmt = 'ro', label = r"P($N\geq2$)")
ax[0][0].plot(mu_model_arr,(Ri_binary_analytic(mu_model_arr,eta_i, lambdas = svd)-Ri_pnr_analytic(mu_model_arr,eta_i, k=k,lambdas = svd)), "-r")#,label = r"P($N\geq2$)$=$P($N>0$)$-$P($N=1$) model")
x,y = 0.0695, .95
ax[0][0].annotate('a)',
            xy=(x,y), xycoords='figure fraction',
            horizontalalignment='left', verticalalignment='top',
            fontsize=14)

x,y = 0.0695, .625
ax[1][0].errorbar(mu_arr,Cs1_total_data_arr/rate,yerr=Cs1_total_err_arr/rate,fmt='bo',label= "Signal 1")
ax[1][0].plot(mu_model_arr,Rsj_analytic(mu_model_arr, eta_s1, lambdas = svd),"-b")#, label = "Signal 1 model")
ax[1][0].errorbar(mu_arr,Cs2_total_data_arr/rate,yerr=Cs2_total_err_arr/rate,fmt='go', label="Signal 2")
ax[1][0].plot(mu_model_arr,Rsj_analytic(mu_model_arr, eta_s2, lambdas = svd),"-g")#, label = "Signal 2 model")
ax[1][0].annotate('b)',
            xy=(x,y), xycoords='figure fraction',
            horizontalalignment='left', verticalalignment='top',
            fontsize=14)

x,y = 0.0695, .305
ax[2][0].errorbar(mu_arr,Cis1_binary_data_arr/rate,yerr=Cis1_binary_err_arr/rate,fmt='bo',label= "Threshold Idler")#"Signal 1 & Threshold Idler")
ax[2][0].plot(mu_model_arr,Risj_binary_analytic(mu_model_arr, eta_s1,eta_i, lambdas = svd),"-b")#, label = "Signal 1 & Threshold Idler model")
ax[2][0].errorbar(mu_arr,Cis1_pnr_data_arr/rate,yerr=Cis1_pnr_err_arr/rate,fmt='go',label="PNR Idler")#label= "Signal 1 & PNR Idler")
ax[2][0].plot(mu_model_arr,Risj_pnr_analytic(mu_model_arr, eta_s1,eta_i, k=k, lambdas = svd),"-g")#, label = "Signal 1 & PNR Idler model")
ax[2][0].annotate('c)',
            xy=(x,y), xycoords='figure fraction',
            horizontalalignment='left', verticalalignment='top',
            fontsize=14)


x,y = 0.585, .95
ax[0][1].errorbar(mu_arr,Cis2_binary_data_arr/rate,yerr=Cis2_binary_err_arr/rate,fmt='bo',label= "Threshold Idler")#, label="Signal 2 & Threshold Idler")
ax[0][1].plot(mu_model_arr,Risj_binary_analytic(mu_model_arr, eta_s2, eta_i, lambdas = svd),"-b")#, label = "Signal 2 & Threshold Idler model")
ax[0][1].errorbar(mu_arr,Cis2_pnr_data_arr/rate,yerr=Cis2_pnr_err_arr/rate,fmt='go',label= "PNR Idler")#,label= "Signal 2 & PNR Idler")
ax[0][1].plot(mu_model_arr,Risj_pnr_analytic(mu_model_arr,eta_s2, eta_i,k=k, lambdas = svd),"-g")#, label = "Signal 2 & PNR Idler model")
ax[0][1].annotate('d)',
            xy=(x,y), xycoords='figure fraction',
            horizontalalignment='left', verticalalignment='top',
            fontsize=14)


x,y = 0.585, .625
ax[1][1].errorbar(mu_arr,Cs1s2_data_arr/rate,yerr=Cs1s2_err_arr/rate,fmt='bo')#,label= "Signal 1 & Signal 2")
ax[1][1].plot(mu_model_arr,Rs1s2_analytic(mu_model_arr, eta_s1, eta_s2, lambdas = svd),"-b")#, label = "Signal 1 & Signal 2 model")
ax[1][1].annotate('e)',
            xy=(x,y), xycoords='figure fraction',
            horizontalalignment='left', verticalalignment='top',
            fontsize=14)

x,y = 0.585, .305
ax[2][1].errorbar(mu_arr,Cis1s2_binary_data_arr/rate,yerr=Cis1s2_binary_err_arr/rate,fmt='bo',label= "Threshold Idler")#,label= "Signal 1 & Signal 2 & Threshold Idler")
ax[2][1].plot(mu_model_arr,Ris1s2_binary_analytic(mu_model_arr, eta_s1, eta_s2,eta_i, lambdas = svd),"-b")#, label = "Signal 1 & Signal 2 model")
ax[2][1].errorbar(mu_arr,Cis1s2_pnr_data_arr/rate,yerr=Cis1s2_pnr_err_arr/rate,fmt='go', label= "PNR Idler")#,label= "Signal 1 & Signal 2 & PNR Idler")
ax[2][1].plot(mu_model_arr,Ris1s2_pnr_analytic(mu_model_arr, eta_s1, eta_s2,eta_i,k=k, lambdas = svd),"-g")#, label = "Signal 1 & Signal 2 model")
ax[2][1].annotate('f)',
            xy=(x,y), xycoords='figure fraction',
            horizontalalignment='left', verticalalignment='top',
            fontsize=14)


for i in range(3):
    for j in range(2):
        ax[i][j].set_xlabel(r"$\mu$", fontsize = 14)
        ax[i][j].set_ylabel(r"Probability", fontsize = 14)
        ax[i][j].set_title(title_arr[i][j], fontsize = 16)
        # if i!=2 and j!=1:
        #     ax[i][j].legend()
fontsize = 10
ax[0][0].legend(fontsize=fontsize)
ax[0][1].legend(fontsize=fontsize)
ax[2][0].legend(fontsize=fontsize)
ax[1][0].legend(fontsize=fontsize)
#ax[1][1].legend()
ax[2][1].legend(fontsize=fontsize)
plt.subplots_adjust(left=0.074, bottom = 0.067, right = 0.988, top = 0.92,wspace = 0.277, hspace = 0.562)








fig, axs = plt.subplots(2,1, num=16, figsize = (6,6),gridspec_kw = {'height_ratios': [3, 1]})#, figsize = (6,8))

ax = axs[0]
ax.errorbar(mu_arr,g2_binary_data_arr,yerr=g2_binary_err_arr,fmt='bo', label = "Threshold")#: "+r"$\eta_{s_1}=$"+"{:.2f}, ".format(eta_s1/2)+r"$\eta_{s_2}=$"+"{:.2f}".format(eta_s2/2)+r", $\eta_i=$"+"{:.2f}".format(eta_i)+", multimode")
ax.errorbar(mu_arr,g2_pnr_data_arr,yerr=g2_pnr_err_arr,fmt='go', label = "PNR")#: "+r"$\eta_{s_1}=$"+"{:.2f}, ".format(eta_s1/2)+r"$\eta_{s_2}=$"+"{:.2f}".format(eta_s2/2)+r", $\eta_i=$"+"{:.2f}".format(eta_i)+", $k=$"+"{:.2f}".format(k)+", multimode")
ax.plot(mu_model_arr,g2_0_binary(mu_model_arr, eta_s1,eta_s2,eta_i, lambdas = svd),"-b")#,label = "Threshold Idler")#, "+r"$\eta_{s1}=$"+"{:.3f}, ".format(eta_s1)+r"$\eta_{s2}=$"+"{:.3f}, ".format(eta_s2)+r"$\eta_{i}=$"+"{:.3f}".format(eta_i_arr[i]))
ax.plot(mu_model_arr,g2_0_pnr(mu_model_arr,eta_s1,  eta_s2,eta_i,k=k, lambdas = svd),"-g")#, label = "PNR Idler"+r", $\eta_{i} = $"+"{:.3f}".format(eta_i))
# ax.plot(mu_model_arr,g2_0_binary(mu_model_arr, eta_s1,eta_s2,eta_i, lambdas = [1]),"--b", label = "Threshold (single-mode)")#,label = "Threshold Idler")#, "+r"$\eta_{s1}=$"+"{:.3f}, ".format(eta_s1)+r"$\eta_{s2}=$"+"{:.3f}, ".format(eta_s2)+r"$\eta_{i}=$"+"{:.3f}".format(eta_i_arr[i]))
# ax.plot(mu_model_arr,g2_0_pnr(mu_model_arr,eta_s1,  eta_s2,eta_i,k=10, lambdas = [1]),"--g", label = "PNR (single-mode)")#": "+r"$\eta_{s_1},\eta_{s_2},\eta_{i}=$"+"{:.2f}".format(0.87)+", $k=$"+"{:d}".format(10))#label = "PNR (multi-mode): "+r"$\eta_{s_1},\eta_{s_2},\eta_{i}=$"+"{:.2f}".format(0.87)+", $k=$"+"{:d}".format(10))
ax.plot(mu_model_arr,g2_0_pnr(mu_model_arr,0.87,  0.87,0.87,k, lambdas = [1]),"-m", label = "PNR (single mode): "+r"$\eta_{s_1}=\eta_{s_2}=\eta_{i}=$"+"{:.2f}".format(0.87)+", $k=$"+"{:.2f}".format(k))
#ax.plot(mu_model_arr,g2_0_pnr(mu_model_arr,0.87,  0.87,0.87,5, lambdas = [1]),"--",color = "darkorange", label = "PNR (single mode): "+r"$\eta_{s_1},\eta_{s_2},\eta_{i}=$"+"{:.2f}".format(0.87)+", $k=$"+"{:d}".format(5))
ax.plot(mu_model_arr,g2_0_pnr(mu_model_arr,0.87,  0.87,0.87,10, lambdas = [1]),"-",color = "red", label = "PNR (single mode): "+r"$\eta_{s_1}=\eta_{s_2}=\eta_{i}=$"+"{:.2f}".format(0.87)+", $k=$"+"{:d}".format(10))
ax.set_xlabel(r"$\mu$", fontsize = 12)
ax.set_ylabel("$g^{2}(0)$", fontsize = 14)
ax.set_ylim([0,1.5])
ax.tick_params(axis='both', which='major', labelsize=10)
ax.tick_params(axis='both', which='minor', labelsize=10)
ax.legend()
handles, labels = ax.get_legend_handles_labels()
order = [2,3,0,1]
ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize = 9, loc = "upper right")
left, bottom, width, height =  0.1525,0.66,0.09,0.26#0.15,0.765,0.12,0.16#0.063,0.56,0.08,0.33
ax2 = fig.add_axes([left, bottom, width, height])
mu_model_arr_inset = np.linspace(0.27,0.53, 1000)
ax2.set_xticks(np.arange(0.3,0.7,0.2))
print("g2 diff: ",g2_binary_data_arr[10],g2_binary_err_arr[10],g2_pnr_data_arr[10],g2_pnr_err_arr[10],g2_binary_data_arr[10]-g2_pnr_data_arr[10], g2_pnr_data_arr[10]/g2_binary_data_arr[10])
ax2.errorbar(mu_arr[10:12],g2_binary_data_arr[10:12],yerr=g2_binary_err_arr[10:12],fmt='bo', ms=3,label = "Threshold")
ax2.errorbar(mu_arr[10:12],g2_pnr_data_arr[10:12],yerr=g2_pnr_err_arr[10:12],fmt='go', ms=3,label = "PNR")
ax2.plot(mu_model_arr_inset,g2_0_binary(mu_model_arr_inset, eta_s1,eta_s2,eta_i, lambdas = svd),"-b")#,label = "Threshold Idler")#, "+r"$\eta_{s1}=$"+"{:.3f}, ".format(eta_s1)+r"$\eta_{s2}=$"+"{:.3f}, ".format(eta_s2)+r"$\eta_{i}=$"+"{:.3f}".format(eta_i_arr[i]))
ax2.plot(mu_model_arr_inset,g2_0_pnr(mu_model_arr_inset,eta_s1,  eta_s2,eta_i,k=k, lambdas = svd),"-g")#, label = "PNR Idler"+r", $\eta_{i} = $"+"{:.3f}".format(eta_i))
ax2.tick_params(axis='both', which='major', labelsize=8)
ax2.tick_params(axis='both', which='minor', labelsize=8)
ax2.set_xlabel(r"$\mu$", fontsize = 9)

#ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
plt.subplots_adjust(left=0.125, right = 0.9)


print("low mu")
print(mu_arr[:5])
print("binary low mu: ")
print(g2_binary_data_arr[:5])

print("pnr low mu: ")
print(g2_pnr_data_arr[:5])

#fig, ax = plt.subplots(1,1)
markersize = 10
linewidth = 5
mu_model_arr = np.logspace(-5,-1.5)
ax = axs[1]
#fig, ax = plt.subplots(1,1, num=20, figsize = (8,6))
ax.annotate('CQNET 2021',
            xy=(.78, .97), xycoords='figure fraction',
            horizontalalignment='left', verticalalignment='top',
            fontsize=12)
ax.annotate('a)',
            xy=(.13, .97), xycoords='figure fraction',
            horizontalalignment='left', verticalalignment='top',
            fontsize=12)
ax.annotate('b)',
            xy=(.13, .3), xycoords='figure fraction',
            horizontalalignment='left', verticalalignment='top',
            fontsize=12)
ax.errorbar(mu_arr[:5],g2_binary_data_arr[:5],yerr=g2_binary_err_arr[:5],fmt='bo', label = "Threshold Idler (multi-mode)")#: "+r"$\eta_{s_1}=$"+"{:.2f}, ".format(eta_s1/2)+r"$\eta_{s_2}=$"+"{:.2f}".format(eta_s2/2)+r", $\eta_i=$"+"{:.2f}".format(eta_i)+", multimode")
ax.errorbar(mu_arr[:5],g2_pnr_data_arr[:5],yerr=g2_pnr_err_arr[:5],fmt='go', label = "PNR Idler (multi-mode)")#: "+r"$\eta_{s_1}=$"+"{:.2f}, ".format(eta_s1/2)+r"$\eta_{s_2}=$"+"{:.2f}".format(eta_s2/2)+r", $\eta_i=$"+"{:.2f}".format(eta_i)+", $k=$"+"{:.2f}".format(k)+", multimode")
ax.plot(mu_model_arr,g2_0_binary(mu_model_arr, eta_s1,eta_s2,eta_i, lambdas = svd),"-b",label = "Threshold Idler (multi-mode): "+r"$\eta_{s_1}=$"+"{:.2f}, ".format(eta_s1/2)+r"$\eta_{s_2}=$"+"{:.2f}".format(eta_s2/2)+r", $\eta_i=$"+"{:.2f}".format(eta_i)+", $k=$"+"{:.2f}".format(k))#, "+r"$\eta_{s1}=$"+"{:.3f}, ".format(eta_s1)+r"$\eta_{s2}=$"+"{:.3f}, ".format(eta_s2)+r"$\eta_{i}=$"+"{:.3f}".format(eta_i_arr[i]))
f0 = lambda x: 0.0071-g2_0_binary(x, eta_s1,eta_s2,eta_i, lambdas = svd)
z =fsolve(f0,0.004)
print("blue: ", z)
ax.plot(z[0]*np.ones(len(mu_model_arr)), np.linspace(0,0.007, len(mu_model_arr)),  color = "lightgrey")

ax.plot(z[0], 0, "xb",clip_on=False,markersize = markersize,linewidth=linewidth)

ax.plot(mu_model_arr,g2_0_pnr(mu_model_arr,eta_s1,  eta_s2,eta_i,k=k, lambdas = svd),"-g",label = "PNR Idler (multi-mode): "+r"$\eta_{s_1}=$"+"{:.2f}, ".format(eta_s1/2)+r"$\eta_{s_2}=$"+"{:.2f}".format(eta_s2/2)+r", $\eta_i=$"+"{:.2f}".format(eta_i)+", $k=$"+"{:.2f}".format(k))#, label = "PNR Idler"+r", $\eta_{i} = $"+"{:.3f}".format(eta_i))
f1 = lambda x: 0.0071-g2_0_pnr(x,eta_s1,  eta_s2,eta_i,k=k, lambdas = svd)
z =fsolve(f1,0.004)
ax.plot(z[0]*np.ones(len(mu_model_arr)), np.linspace(0,0.007, len(mu_model_arr)),  color = "lightgrey")
ax.plot(z[0], 0, "xg",clip_on=False,markersize = markersize,linewidth=linewidth)
print("green: ", z)



ax.plot(mu_model_arr,g2_0_pnr(mu_model_arr,0.87,  0.87,0.87,k, lambdas = [1]),"-m", label = "PNR Idler (multi-mode): "+r"$\eta_{s_1},\eta_{s_2},\eta_{i}=$"+"{:.2f}".format(0.87)+", $k=$"+"{:d}".format(10))
f2 = lambda x: 0.007-g2_0_pnr(x,0.87,  0.87,0.87,k, lambdas = [1])
z =fsolve(f2,0.004)
print("purple: ", z)
ax.plot(z[0]*np.ones(len(mu_model_arr)), np.linspace(0,0.007, len(mu_model_arr)), color = "lightgrey")
ax.plot(z[0], 0, "x", color = "m", clip_on=False,markersize = markersize,linewidth=linewidth)

ax.plot(mu_model_arr,g2_0_pnr(mu_model_arr,0.87,  0.87,0.87,10, lambdas = [1]),"-r", label = "PNR Idler (single mode): "+r"$\eta_{s_1},\eta_{s_2},\eta_{i}=$"+"{:.2f}".format(0.87)+", $k=$"+"{:d}".format(10))
f3 = lambda x: 0.007-g2_0_pnr(x,0.87,  0.87,0.87,10, lambdas = [1])
z =fsolve(f3,0.004)
print("red: ", z)
print("mu at higher efficiency", z)
ax.plot(z[0]*np.ones(len(mu_model_arr)), np.linspace(0,0.007, len(mu_model_arr)),  color = "lightgrey")
ax.plot(z[0], 0, "x", color = "r", clip_on=False, markersize = markersize,linewidth=linewidth)

ax.plot(mu_model_arr, 0.007*np.ones(len(mu_model_arr)),linestyle="--",color  = "k")

ax.set_xlabel(r"$\mu$", fontsize = 12)
ax.set_ylabel("$g^{2}(0)$", fontsize = 14)#, rotation = 90)
ax.set_xlim([0.0001,0.015])
ax.set_ylim([0.00,0.012])
ax.tick_params(axis='both', which='major', labelsize=10)
ax.tick_params(axis='both', which='minor', labelsize=10)


ax.set_xticks(np.arange(0,0.020,0.005))
ax.set_yticks(np.arange(0,0.014,0.002))
for i, label in enumerate(ax.yaxis.get_ticklabels()):
    if i%2!=0:
        label.set_visible(False)
plt.subplots_adjust(left = 0.133, bottom = 0.086, right = 0.955, top = 0.935, wspace = 0.2, hspace = 0.351)

class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format


# ax.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
# ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,-3))


# ax.ticklabel_format(axis="y", style="sci", scilimits=(1,0))



plt.show()
