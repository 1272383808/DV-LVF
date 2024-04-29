from sklearn.cluster import KMeans
from torch import nn

from run import *

# 读取数据文件
hidden_dim = 768
vul_hidden = 192
m = 10
file_path = '../data/train.csv'
model_path = 'D:\LLMs\codet5-base'
df_all = pd.read_csv(file_path)

df_vul = df_all[df_all["function_label"] == 1].reset_index(drop=True)

vulIs = df_vul["vul_patterns"].tolist()

low = nn.Linear(hidden_dim, vul_hidden)
rnn_statement_embedding = nn.GRU(input_size=hidden_dim,
                                              hidden_size=hidden_dim,
                                              num_layers=1,
                                              batch_first=True)
rnn_vulI_pooling = nn.GRU(input_size=hidden_dim,
                                            hidden_size=hidden_dim,
                                            num_layers=1,
                                            batch_first=True)

tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
t5 = T5EncoderModel.from_pretrained(model_path)
word_embedding = t5.shared

vulset = torch.zeros(len(vulIs), vul_hidden)
for i in range(len(vulIs)):
    vulI = vulIs[i].split("\n")
    vulI = vulI[:m]
    padding_statement = [tokenizer.pad_token_id for _ in range(25)]
    input_ids = []
    for stat in vulI:
        ids_ = tokenizer.encode(str(stat),
                                truncation=True,
                                max_length=25,
                                padding='max_length',
                                add_special_tokens=False)
        input_ids.append(ids_)
    if len(input_ids) < m:
        for _ in range(m-len(input_ids)):
            input_ids.append(padding_statement)
    embed = word_embedding(input_ids)
    vulIEmbed = low(rnn_vulI_pooling(rnn_statement_embedding(embed)))
    vulset[i] = vulIEmbed

data_np = vulset.detach().numpy()

# 使用KMeans算法聚类数据到150个簇中心
kmeans = KMeans(n_clusters=150)
kmeans.fit(data_np)

# 获取簇中心
cluster_centers = kmeans.cluster_centers_

# 将簇中心转换为PyTorch张量
cluster_centers_tensor = torch.from_numpy(cluster_centers)

# 保存
torch.save(cluster_centers_tensor, "vuldict.pth")
# 检查簇中心的大小
print(cluster_centers_tensor.size())  # [150, 192]