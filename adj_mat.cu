using namespace std;
#include <vector>
#include <map>

int DEFAULT_BETTING_SIZE = 100
// Mapping values
map<int, string> BETTING_ACTION = {
    {0, BET}, 
    {1, CALL},
    {2, FOLD},
    {3, CHECK},
    {4, RAISE}
};

// GameState matrix
// Row as players
// Column as betting size, betting action, current pot size at each player's turn, community cards, hole cards
// Mapping: numerical value to abstracted cards, numerical value to set of actions
// TODO create numerical mapping for abstracted cards

// CPU-side
class GameStateMatrix {
    public:
        // The game state matrix itself
        // Internal representation as stacked row matrix
        vector<vector<int>> gameStateMatrix;
        
        // Function to add player; numbered by their index
        __host__ void addPlayer(int num) {
            for (int i = 0; i < num; i++) {
                // Row as player and their info
                vector<int> player = {DEFAULT_BETTING_SIZE, -1, 0, -1, -1};
                gameStateMatrix.push_back(player);
            };
        }

        // Function to parse the community cards and hole cards
        __host__ void parseCard() {
            
        }
}