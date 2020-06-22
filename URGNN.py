import numpy as np
import pandas as pd
import pickle
import csv
import os
import torch
import csv
from torch_geometric.data import Data
from tqdm import tqdm
np.random.seed(42)
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import DataLoader

from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('ratings.csv', dtype = {"userId": int, "movieId": int, "rating": float, "timestamp": str})
df.columns=['userId','movieId','rating','timestamp']


#filter out item session with Length < 2
df['valid_user'] = df.userId.map(df.groupby('userId')['movieId'].size() > 2 )
df = df.loc[df.valid_user].drop('valid_user', axis=1)

#filter out  user200 ~ ...
# df['valid_user2'] = df.userId.map(df.groupby('userId')['userId'].value() > 200)
# df = df.loc[df.valid_user2].drop('valid_user2', axis=1)

# #randomly sample a couple of them
#sampled_session_id = np.random.choice(df.userId.unique(), 8000, replace=False)
sampled_session_id = np.random.choice(df.userId.unique(), 8000, replace=False)
df = df.loc[df.userId.isin(sampled_session_id)]

positive_df = df

#encder part
user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()
df['userId'] = user_encoder.fit_transform(df.userId )
df['movieId']= movie_encoder.fit_transform(df.movieId)
positive_df['userId'] = user_encoder.fit_transform(positive_df.userId )
positive_df['movieId']= movie_encoder.fit_transform(positive_df.movieId)

#rating > 3  filter
positive_df = positive_df.loc[positive_df['rating']>3.0]

#for answer label
#rating > 3
#dictionary
positive_movie_dict = dict(positive_df.groupby('userId')['movieId'].apply(list))





class YooChooseDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(YooChooseDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['../input/moive.dataset']

    def download(self):
        pass

    def process(self):

        data_list = []

        # process by session_id
        grouped = df.groupby('userId')
        for userId, group in tqdm(grouped):
            le = LabelEncoder()
            user_moive_id = le.fit_transform(group.movieId)
            group = group.reset_index(drop=True)
            group['user_moive_id'] = user_moive_id
            node_features = \
                group.loc[group.userId == userId, ['user_moive_id','userId','movieId']].sort_values(
                    'user_moive_id')[['userId','movieId']].drop_duplicates().values

            node_features = torch.LongTensor(node_features).unsqueeze(1)
            target_nodes = group.user_moive_id.values[1:]
            source_nodes = group.user_moive_id.values[:-1]
            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

            x = node_features


            if userId in positive_movie_dict:
                positive_indices = le.transform(positive_movie_dict[userId])
                label = np.zeros(len(node_features))
                label[positive_indices] = 1
            else:
                label = [0] * len(node_features)


            y = torch.FloatTensor(label)


            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])






dataset = YooChooseDataset('./')

dataset = dataset.shuffle()
one_tenth_length = int(len(dataset) * 0.1)
train_dataset = dataset[:one_tenth_length * 8]  # 80% of entire dataset
val_dataset = dataset[one_tenth_length*8:one_tenth_length * 9]  # 10% of entire dataset
test_dataset = dataset[one_tenth_length*9:]   # 10% of entire dataset
#print(len(train_dataset), len(val_dataset), len(test_dataset))

print("train_dataset=",  train_dataset)
print("val_dataset=",  val_dataset)
print("test_dataset=",  test_dataset)

batch_size= 256 #512
train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

num_items = df.userId.max() +1
num_categories = df.movieId.max()+1
#print(num_items , num_categories)

embed_dim = 128
from torch_geometric.nn import GraphConv, TopKPooling, GatedGraphConv, SAGEConv, SGConv,ARMAConv, GATConv,SAGPooling
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = GraphConv(embed_dim * 2, 128)
        self.pool1 = TopKPooling(128, ratio=0.9)

        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.9)

        self.conv3 = GraphConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.9)

        self.item_embedding = torch.nn.Embedding(num_embeddings=num_items, embedding_dim=embed_dim)
        self.category_embedding = torch.nn.Embedding(num_embeddings=num_categories, embedding_dim=embed_dim)

        self.lin1 = torch.nn.Linear(256, 256)
        self.lin2 = torch.nn.Linear(256, 128)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # print('x =', x)
        # print('====================')

        item_id = x[:, :, 0]
        category = x[:, :, 1]

        # print("item_id: ", item_id)
        # print("category: \n", category)
        # print('====================')


        emb_item = self.item_embedding(item_id).squeeze(1)
        emb_category = self.category_embedding(category).squeeze(1)

        #         emb_item = emb_item.squeeze(1)
        #         emb_cat
        x = torch.cat([emb_item, emb_category], dim=1)
        #         print(x.shape)
        x = F.relu(self.conv1(x, edge_index))
        #                 print(x.shape)
        x, edge_index, _, batch, _ , _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))

        x, edge_index, _, batch, _ , _= self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))

        x, edge_index, _, batch, _ , _= self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.act2(x)

        outputs = []
        for i in range(x.size(0)):
            output = torch.matmul(emb_item[data.batch == i], x[i, :])

            outputs.append(output)

        x = torch.cat(outputs, dim=0)
        x = torch.sigmoid(x)

        # save
        # savePath = "./output/test_model.pth"
        # torch.save(model.state_dict(), savePath)


        return x




device = torch.device('cuda')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
crit = torch.nn.BCELoss()



def train():
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)

        label = data.y.to(device)
        loss = crit(output, label)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()





    #
    # f = open('./output/weights.csv','w',newline='')
    # wr = csv.writer(f)
    # wr.writerows(model.item_embedding.weight)
    # f.close()
    #     for Aweight in model.item_embedding.weight:
    #         wr.writerow(Aweight)
    #
    # f.close()

    return loss_all / len(train_dataset)


from sklearn.metrics import roc_auc_score

def evaluate(loader):
    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data).detach().cpu().numpy()

            label = data.y.detach().cpu().numpy()
            predictions.append(pred)
            labels.append(label)

    predictions = np.hstack(predictions)
    labels = np.hstack(labels)

    return roc_auc_score(labels, predictions)

for epoch in range(1, 200):
    loss = train()

    if epoch == 10:
        # print(model.item_embedding.weight)
        with open('./output/weights10.csv', 'w', newline='') as f:
            writer = csv.writer((f))
            for Aweight in model.item_embedding.weight:
                writer.writerow(Aweight.tolist())

    if epoch == 50:
        # print(model.item_embedding.weight)
        with open('./output/weights50.csv', 'w', newline='') as f:
            writer = csv.writer((f))
            for Aweight in model.item_embedding.weight:
                writer.writerow(Aweight.tolist())

    if epoch == 100:
        # print(model.item_embedding.weight)
        with open('./output/weights100.csv', 'w', newline='') as f:
            writer = csv.writer((f))
            for Aweight in model.item_embedding.weight:
                writer.writerow(Aweight.tolist())

    #train_acc = evaluate(train_loader)
    #val_acc = evaluate(val_loader)
    test_acc = evaluate(test_loader)
    #print('Epoch: {:03d}, Loss: {:.5f}, Train Auc: {:.5f}, Val Auc: {:.5f}, Test Auc: {:.5f}'.format(epoch, loss, train_acc, val_acc, test_acc))
    print('Epoch: {:03d}, Loss: {:.5f}, Test Auc: {:.5f}'.format(epoch, loss,test_acc))











