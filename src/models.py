import numpy as np

class LogisticRegression:
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization='l2', lambda_reg=0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
        self.losses = []
        self.train_accuracies = []
        
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def compute_loss(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        if self.regularization == 'l2':
            loss += (self.lambda_reg / (2 * len(y_true))) * np.sum(self.weights**2)
        return loss
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.losses = []
        self.train_accuracies = []
        
        for i in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
            error = y_pred - y
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)
            if self.regularization == 'l2':
                dw += (self.lambda_reg / n_samples) * self.weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            if i % 100 == 0:
                loss = self.compute_loss(y, y_pred)
                self.losses.append(loss)
                preds = (y_pred >= 0.5).astype(int)
                acc = np.mean(preds == y)
                self.train_accuracies.append(acc)
    
    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)
    
    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

class KNN:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict_one(x) for x in X]
        return np.array(y_pred)
    
    def predict_proba(self, X):
        y_proba = [self._predict_proba_one(x) for x in X]
        return np.array(y_proba)

    def _predict_one(self, x):
        distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices].astype(int)
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common
    
    def _predict_proba_one(self, x):
        distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        prob_positive = np.sum(k_nearest_labels == 1) / self.k
        return prob_positive

def find_best_k(X_train, y_train, X_val, y_val, max_k=20):
    accuracies = []
    k_values = range(1, max_k + 1)
    
    for k in k_values:
        model = KNN(k=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        accuracies.append(acc)
        
    best_acc = max(accuracies)
    best_k = k_values[np.argmax(accuracies)]
    return best_k, best_acc, accuracies

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def f1_score(y_true, y_pred):
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])

def roc_curve(y_true, y_scores):
    thresholds = np.sort(np.unique(y_scores))[::-1]
    tpr_list = []
    fpr_list = []
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        return np.array([0., 1.]), np.array([0., 1.])
    
    for thresh in thresholds:
        y_pred_temp = (y_scores >= thresh).astype(int)
        tp = np.sum((y_true == 1) & (y_pred_temp == 1))
        fp = np.sum((y_true == 0) & (y_pred_temp == 1))
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)
        
    return np.array(fpr_list), np.array(tpr_list)

def auc_score(fpr, tpr):
    if fpr.shape[0] < 2: return 0.5
    direction = -1 if fpr[0] > fpr[-1] else 1
    return direction * np.trapz(tpr, fpr)

def cross_validation(X, y, model_class, k_folds=5, **model_params):
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    fold_indices = np.array_split(indices, k_folds)
    scores = []
    
    for i in range(k_folds):
        test_idx = fold_indices[i]
        train_idx = np.concatenate([fold_indices[j] for j in range(k_folds) if j != i])
        
        X_train_cv, X_test_cv = X[train_idx], X[test_idx]
        y_train_cv, y_test_cv = y[train_idx], y[test_idx]
        
        model = model_class(**model_params)
        model.fit(X_train_cv, y_train_cv)
        
        y_pred_cv = model.predict(X_test_cv)
        acc = accuracy_score(y_test_cv, y_pred_cv)
        scores.append(acc)
    
    return np.array(scores), np.mean(scores), np.std(scores)