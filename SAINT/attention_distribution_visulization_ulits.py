import seaborn as sns
sns.set_style("white")
#from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Colormap

def showDistribution(attention_distribution, protein_number=0, query=0):
  x = list(range(700))
  plt.plot(x, attention_distribution[protein_number][query])
  plt.show()


plt.switch_backend('agg')


def showPlot(points):
  plt.figure()
  fig, ax = plt.subplots()
  loc = ticker.MultipleLocator(base=0.2)
  ax.yaxis.set_major_locator(loc)
  plt.plot(points)


def showAttention(input_sequence, attentions, Total_Protein_Map=True, amino_acid_number=0):
  dim = 20
  fig = plt.figure(figsize=(dim, dim))
  ax = fig.add_subplot(111)
  if Total_Protein_Map:
    cax = ax.matshow(attentions, cmap='Blues')
    # Possible values for 'cmap' are: Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cividis, cividis_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, icefire, icefire_r, inferno, inferno_r, jet, jet_r, magma, magma_r, mako, mako_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, rocket, rocket_r, seismic, seismic_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r...
  else:
    cax = ax.matshow(attentions[amino_acid_number:amino_acid_number + 1], cmap='Blues')
    att_show_for_single_query = [round(float((x / .09) * 255)) for x in attentions[amino_acid_number]]
    print('protein 37, query number 2:\n', att_show_for_single_query)

  fig.colorbar(cax)

  SMALL_SIZE = 8
  MEDIUM_SIZE = 10
  BIGGER_SIZE = 50

  plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
  plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
  plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
  plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
  plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
  plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
  plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

  ax.set_xticklabels([''] + input_sequence)
  if Total_Protein_Map:
    ax.set_yticklabels([''] + input_sequence)
  else:
    ax.set_yticklabels(['', ''])  # + input_sequence)

  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

  plt.show()


def evaluateAndShowAttention(input_sequence, attention_matrix, protein_number, lengths, number_to_name_map,
                             Total_Protein_Map=True, amino_acid_number=0):
  # print('input =', input_sequence[protein_number])
  # print('input =', input_sequence[protein_number])
  length = lengths[protein_number]
  input_seq = [number_to_name_map[i] for i in input_sequence[protein_number][:length]]
  print(input_seq)
  showAttention(input_seq, attention_matrix[protein_number][:length, :length], Total_Protein_Map=Total_Protein_Map,
                amino_acid_number=amino_acid_number)


number_to_aminoAcidName_map = {
  0: 'A',
  1: 'C',
  2: 'E',
  3: 'D',
  4: 'G',
  5: 'F',
  6: 'I',
  7: 'H',
  8: 'K',
  9: 'M',
  10: 'L',
  11: 'N',
  12: 'Q',
  13: 'P',
  14: 'S',
  15: 'R',
  16: 'T',
  17: 'W',
  18: 'V',
  19: 'Y',
  20: 'X',
  21: 'NoSeq'
}
number_to_prediction_map = {
  0: 'L',
  1: 'B',
  2: 'E',
  3: 'G',
  4: 'I',
  5: 'H',
  6: 'S',
  7: 'T',
  8: 'NoSeq'
}
