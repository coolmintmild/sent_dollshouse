import pandas as pd
import matplotlib.pyplot as plt
import os

# Calculate moving average and draw a graph
def draw_MA_graph(df, window_ratio):
    # Compute window size based on ratio
    window_size = int(len(df) * window_ratio)

    # Calculate moving average
    df['moving_avg'] = df['compound'].rolling(window=window_size).mean()

    # Draw the graph
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['moving_avg'], label='Moving Average')
    plt.xlabel('Index')
    plt.ylabel('Moving Average')
    plt.title('Moving Average of Compound Scores')
    plt.legend()
    plt.grid(True)

    # Ensure the output directory exists
    os.makedirs(os.path.join(root_path, 'graphs'), exist_ok=True)

    # Save the graph
    plt.savefig(f"{root_path}/graphs/MA-graph_{window_ratio}.png")

    # Show the graph
    plt.show()

# Calculate moving average and draw a graph for two characters
def draw_MA_graph_indv(df, window_ratio, character1, character2):
    # Filter dataframes for each character
    df1 = df[df['character'] == character1]
    df2 = df[df['character'] == character2]

    # Compute window size based on ratio
    window_size1 = int(len(df1) * window_ratio)
    window_size2 = int(len(df2) * window_ratio)

    # Calculate moving average for each character
    df1 = df1.assign(moving_avg=df1['compound'].rolling(window=window_size1, center=True).mean())
    df2 = df2.assign(moving_avg=df2['compound'].rolling(window=window_size2, center=True).mean())

    # Draw the graph
    plt.figure(figsize=(14, 7))
    plt.plot(df1['index'], df1['moving_avg'], label=f'Moving Average - {character1}')
    plt.plot(df2['index'], df2['moving_avg'], label=f'Moving Average - {character2}')
    plt.xlabel('Index')
    plt.ylabel('Moving Average')
    plt.title('Moving Average of Compound Scores')
    plt.legend()
    plt.grid(True)

    # Ensure the output directory exists
    os.makedirs(os.path.join(root_path, 'graphs'), exist_ok=True)

    # Save the graph
    plt.savefig(f"{root_path}/graphs/MA-graph_{window_ratio}_{character1}_{character2}.png")

    # Show the graph
    plt.show()

root_path = '/content/drive/MyDrive/nlp/sentiment'
file_path = f"{root_path}/SentbySent_scored.xlsx"

# Load the dataframe
df = load_data(file_path)
