import pickle
import numpy as np
from tqdm import tqdm
import pandas as pandas
from umap.umap_ import UMAP
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler

df = pandas.read_csv("../csvs/Hecktor22_AugmentedRadiomics.csv")

# Some feature output complex values, this might be caused by specific transforms...
# To investigate :)
for column in df.columns:
    if column.split("_")[-1] in ["Elongation", "Flatness", "LeastAxisLength", "MinorAxisLength", "MajorAxisLength"]:
        df[column] = df[column].apply(lambda x: np.real(np.complex64(x)))

endpoints = pandas.read_csv("../csvs/hecktor2022_endpoint_training.csv")
df["Patient Name"] = df["Patient ID"].apply(lambda x: x.split("_")[0])


df = pandas.merge(df, endpoints, left_on="Patient Name", right_on="PatientID")

umap = UMAP(n_components=2)


#df = df.fillna(0)
df = df.dropna()
# Remove Target columns
df = df.drop("Relapse", axis=1)
df = df.drop("PatientID", axis=1)
df = df.drop("Patient Name", axis=1)
#df = df.drop("Unnamed: 0", axis=1)
vecs = []
trues = []
paragraphs = []
colors = []
colors_true = []
for id_ in tqdm(df["Patient ID"]):
    vec = df[df["Patient ID"]==id_]
    is_original = "Identity_Identity" in id_
    if "ContrastShift" in id_ or "Gamma" in id_ or "Dilate4mm" in id_:
        pass
    else:
        vec = vec.drop("Patient ID", axis=1)
        if is_original:
            trues.append([list(map(float, i)) for i in vec.values][0])
            colors_true.append(vec["RFS"])
        vecs.append([list(map(float, i)) for i in vec.values][0])
        paragraphs.append(id_)
        colors.append(vec["RFS"])
df = df.drop("RFS", axis=1)



thresh = np.inf
close_vecs = []
close_rfs = []
for i, vec in tqdm(enumerate(vecs)):
    feat_num = 0
    true = trues[int(i//(len(vecs)/len(trues)))]
    if  distance.euclidean(np.array(vec), np.array(true), abs(1 / (np.array(true) + .1))) / len(vec) < thresh:
        close_vecs.append(vec)
        close_rfs.append(colors[i])
print(np.shape(close_vecs))

umap.fit(trues)
umap = pickle.load(open("umap.pkl", "rb"))

out_t = umap.transform(trues)
out = umap.transform(close_vecs)
print(np.shape(out))
#pickle.dump(umap, open("umap.pkl", "wb"))

fig,ax = plt.subplots()

sc = plt.scatter(out[:,0], out[:,1], c=close_rfs, s=np.ones_like(close_rfs))

plt.scatter(out_t[:,0], out_t[:,1], c=colors_true, s=[i*12 for i in np.ones_like(colors_true)])

annot = ax.annotate("", xy=(0,0), xytext=(10,10),textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
ax.legend()
#annot.set_visible(False)

def update_annot(ind):
    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    #text = "{}, {}".format(" ".join(list(map(str, ind["ind"]))),
    #                       " ".join([paragraphs[n] for n in ind["ind"]]))
    text = " ".join([paragraphs[n] for n in ind["ind"]])

    annot.set_text(text)
    # annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
    annot.get_bbox_patch().set_alpha(0.4)

def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)
fig.set_size_inches(76.8*2,43.2*2)

plt.show()
