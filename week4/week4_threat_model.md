# Part C: Privacy Threats in Federated Learning

* `Honest-but-curious server`：服务器 **按照协议正常执行训练，却会分析收到的信息，尝试了解客户端的私有数据** 。例如，它正常下发模型、执行 FedAvg，但同时观察客户端的参数更新，猜测这个客户端主要拥有哪些数字。
* `Malicious client`：主动违反训练规则、试图达到攻击目的的客户端。例如，一个客户端可能故意上传伪造参数，让全局模型产生错误预测；也可能操纵自己的训练行为，帮助推断其他客户端的数据。
* `Model update leakage`：指 **客户端上传的模型参数或参数变化，暴露了训练数据中的信息** 。
* `Gradient inversion`：指 **攻击者利用梯度，尝试反推出产生这些梯度的原始训练数据** 。
* `Membership inference`：指 **判断某个具体样本是否参与过模型训练** 。
* `Property inference`：指 **推断客户端训练数据具有哪些属性，而不一定定位某个具体样本** 。
* `Data minimization`：指 **只收集、传输和保存完成任务所必需的信息** 。例如，仅执行 FedAvg 参数聚合时，服务器需要模型更新和聚合权重信息，没有必要额外收集原始图片、个人身份信息或每张图片的特征向量。
* `Threat model`：威胁模型是 **对攻击场景和攻击者能力的明确假设** 。

## What information is transmitted in your FedAvg implementation?

The server sends the global model’s weights and biases to each participating client. Each client returns its locally trained weights and biases, along with its training sample count for weighted aggregation.

## What information could a server potentially infer from client updates?

A server may infer properties of a client’s data, whether a specific sample was used for training, and, under certain conditions, reconstruct original training samples.

## How does the risk change when a client has very few local samples?

The privacy risk may increase because each sample has a larger influence on the client update, making individual training samples easier to infer or reconstruct.

## What assumptions are made about the server and clients?

We assume an honest-but-curious server that follows the protocol but may analyze client updates to infer private information. Clients are assumed to follow the protocol and provide genuine updates and sample counts, while keeping raw data local.

## Why is “raw data are not uploaded” not equivalent to “the system is private”?

Even if raw data are not uploaded, the server may infer sensitive information or reconstruct training samples from the shared model parameters or updates. Therefore, keeping raw data local does not guarantee privacy.
