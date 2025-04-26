#include<iostream>
#include<queue>
#include<omp.h>
using namespace std;

// Define the structure for a node
class Node {
public:
    int data;
    Node *left, *right;
};

// Function to insert nodes level-by-level
Node* insert(Node* root, int data) {
    if (!root) {
        root = new Node{data, NULL, NULL};
        return root;
    }

    queue<Node*> q;
    q.push(root);

    while (!q.empty()) {
        Node* temp = q.front();
        q.pop();

        if (!temp->left) {
            temp->left = new Node{data, NULL, NULL};
            return root;
        } else {
            q.push(temp->left);
        }

        if (!temp->right) {
            temp->right = new Node{data, NULL, NULL};
            return root;
        } else {
            q.push(temp->right);
        }
    }
    return root;
}

// Parallel BFS Traversal
void bfsParallel(Node* root) {
    if (!root) return;

    queue<Node*> q;
    q.push(root);

    while (!q.empty()) {
        int levelSize = q.size();
        vector<Node*> currentLevel;

        // Collect all nodes at the current level
        for (int i = 0; i < levelSize; i++) {
            currentLevel.push_back(q.front());
            q.pop();
        }

        // Parallel traversal of current level
        #pragma omp parallel for
        for (int i = 0; i < currentLevel.size(); i++) {
            Node* curr = currentLevel[i];

            #pragma omp critical
            cout << curr->data << " ";

            #pragma omp critical
            {
                if (curr->left) q.push(curr->left);
                if (curr->right) q.push(curr->right);
            }
        }
    }
}

int main() {
    Node* root = NULL;
    int data;
    char choice;

    do {
        cout << "Enter data: ";
        cin >> data;

        root = insert(root, data);

        cout << "Add more? (y/n): ";
        cin >> choice;

    } while (choice == 'y' || choice == 'Y');

    cout << "\nParallel BFS traversal: ";
    bfsParallel(root);

    return 0;
}
