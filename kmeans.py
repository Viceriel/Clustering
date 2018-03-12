class KMeans:
    """K-Means algorithm implementation"""

    def __init__(self, data, centers):
        """Initialization function, centoids generation"""
        self.data = data

        import random
        from datetime import datetime
        random.seed(datetime.now())
        center = 1

        self.centroids = []
        self.centroids.append(self.data[255][255])

        while center < centers:
            x = random.randint(0, 511)
            y = random.randint(0, 511)
            color = self.data[x][y]
            if not color in self.centroids and color != 0:
                self.centroids.append(color)
                center = center + 1

        print(self.centroids)

    def clustering(self, tolerance):
        bins = len(self.centroids)
        colors = []
        for i in range(0, bins):
            colors.append([])

        for i in range(0, 511):
            for j in range(0,511):
                color = self.data[i][j]
                if color != 0:
                    ind = self.centerAssigne(color, bins)
                    colors[ind].append(color)

        import numpy as np
        for i in range(0, bins):
            self.centroids[i] = np.mean(colors[i])

        return self.printImg()


    def centerAssigne(self, color, centers):
        min = abs(color-self.centroids[0])
        index = 0

        for i in range(1, centers):
            diff = abs(color-self.centroids[i])
            if diff < min:
                min = diff
                index = i

        return index

    def printImg(self):
        import numpy as np
        img = np.ones((512,512, 3))

        for i in range(0, 511):
            for j in range(0, 511):
                if self.data[i][j] == 0:
                    img[i, j, 0] = 0
                else:
                    ind = self.centerAssigne(self.data[i][j], 3)
                    if ind == 0:
                        img[i,j, 0] = 255
                    elif ind == 1 :
                        img[i,j, 1] = 255
                    else:
                        img[i,j, 2]= 255

        return img
