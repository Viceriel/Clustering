class KMeans:
    """K-Means algorithm implementation"""

    def __init__(self, data, centers):
        """Initialization function, centroids generation"""
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

    def avginit(self, bins):
        self.averages = []
        for i in range(0, bins):
            self.averages.append(0)

    def vecdiff(self,tolerance,bins):
        sum = 0
        for i in range(0,bins):
            sum = sum + abs(self.averages[i]-self.centroids[i])
        if sum > tolerance:
            return 1
        return 0


    def clustering(self, tolerance):
        """Perform cluster analysis about provided data"""
        import matplotlib.pyplot as plt
        bins = len(self.centroids)
        self.avginit(bins)

        while 1:            
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
                self.averages[i] = np.mean(colors[i])
                print(len(colors[i]))
            if self.vecdiff(tolerance, bins):
                for i in range(0, bins):
                    self.centroids[i] = self.averages[i]
                
                print(self.centroids)
            else:
                 for i in range(0, bins):
                    self.centroids[i] = self.averages[i]
                    return self.printImg()

            #plt.figure()
            #plt.ion()
            #plt.imshow(self.printImg())
            #plt.colorbar()
            #plt.show() 
            #plt.pause(0.001)
        return self.printImg()


    def centerAssigne(self, color, centers):
        """Computing distance between data and centroids"""
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
        img = np.zeros((512,512))

        for i in range(0, 511):
            for j in range(0, 511):
                if self.data[i][j] == 0:
                    img[i][j] = 0
                else:
                    ind = self.centerAssigne(self.data[i][j], 3)
                    img[i][j] = self.centroids[ind]
        return img
