import pyspiel
game = pyspiel.load_game("universal_poker(betting=nolimit,numPlayers=2,numRounds=4,blind=2 1,firstPlayer=2 1 1 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1,stack=200 200,bettingAbstraction=fcpa)")
state = game.new_initial_state()
print(state.to_string())
