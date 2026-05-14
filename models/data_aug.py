"""
Parker Loupessis Final Project
Class to create more data through data augmentation
"""
import torch
import random

class Route_Augmentation:
    def __init__(self, target_counts):
        self.target_counts = target_counts

    def change_side(self, route, length):
        aug_route = route.clone()
        aug_route[:length, [1,3]] = 53.3 - aug_route[:length, [1,3]] # flip y coords and snap_y
        aug_route[:length, [5,7]] = -aug_route[:length, [5,7]] # flip relative y coords and dir_sin 
        return aug_route
    
    def change_direction(self, route, length):
        aug_route = route.clone()
        aug_route[:length, [0,2]] = 120 - aug_route[:length, [0,2]] # flip x coords and snap_x
        aug_route[:length, [4,8]] = -aug_route[:length, [4,8]] # flip relative x coords and dir_cos
        return aug_route

    def inc_depth(self, route, length):
        factor = random.choice([1.05, 1.1, 1.15])
        aug_route = route.clone()
        for frame in range(1, length):
            aug_route[frame, [0,1,4,5]] = ((route[frame, [0,1,4,5]] - route[frame-1, [0,1,4,5]]) * factor) + aug_route[frame-1, [0,1,4,5]]
            if not ((0 < aug_route[frame, 0] < 120) and (0 < aug_route[frame, 1] < 53.3)):
                return None
        aug_route[:length, [6,9]] *= factor
        return aug_route
    
    def dec_depth(self, route, length):
        factor = random.choice([0.85, 0.9, 0.95])
        aug_route = route.clone()
        for frame in range(1, length):
            aug_route[frame, [0,1,4,5]] = ((route[frame, [0,1,4,5]] - route[frame-1, [0,1,4,5]]) * factor) + aug_route[frame-1, [0,1,4,5]]
            if not ((0 < aug_route[frame, 0] < 120) and (0 < aug_route[frame, 1] < 53.3)):
                return None
        aug_route[:length, [6,9]] *= factor
        return aug_route
    
    def augment(self, X, y):
        X_aug = X.clone()
        y_aug = y.clone()
        current_counts = torch.bincount(y)
        needed = self.target_counts - current_counts
        
        for idx, need in enumerate(needed):
            if need <= 0:
                continue

            route_idx = (y == idx)
            route_X = X[route_idx]
            route_y = y[route_idx]

            perm = torch.randperm(len(route_y))
            perm_idx = 0
            while need > 0:
                if perm_idx >= len(perm):
                    perm = torch.randperm(len(route_y))
                    perm_idx = 0
                
                route = route_X[perm[perm_idx].item()]
                perm_idx += 1
                length = int(route[0,10].item())

                if random.random() < 0.25:
                    if random.random() < 0.5:
                        aug1 = random.choice([self.change_side, self.change_direction])
                        aug2 = random.choice([self.inc_depth, self.dec_depth])
                        aug_route = aug1(route, length)
                        aug_route = aug2(aug_route, length)
                    else:
                        aug_route = self.change_side(route, length)
                        aug_route = self.change_direction(aug_route, length)
                else:
                    aug = random.choice([self.change_side, self.change_direction, self.inc_depth, self.dec_depth])
                    aug_route = aug(route, length)
                
                if aug_route is not None:
                    X_aug = torch.cat([X_aug, aug_route.unsqueeze(0)], dim=0)
                    y_aug = torch.cat([y_aug, torch.tensor([idx])])
                    need -= 1
        return X_aug, y_aug