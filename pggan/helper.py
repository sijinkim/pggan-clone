def get_channel_size(max_channels, min_channels):

    assert max_channels % 2 == 0 and min_channels % 2 == 0

    channel_list = [(max_channels, max_channels)] * 4

    channel = max_channels
    while channel >= min_channels:
        if channel == min_channels:
            channel_list.append((channel, 3))
            channel = 3

        else:
            channel_list.append((channel, channel//2))
            channel = channel//2

    return channel_list
