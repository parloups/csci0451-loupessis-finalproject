"""
Parker Loupessis Final Project
Route Augmentation Class
"""
import torch
import random

class Route_Augmentation:
    def __init__(self, target_counts):
        """
        Args:
            target_counts: int, the number of samples to augment each class up to.
                          Classes already at or above this count are left unchanged.
        """
        self.target_counts = target_counts

    def change_side(self, route, length):
        """
        Mirrors the route across the center of the field (sideline to sideline).
        A route run from the left side of the field becomes the same route on the right.
        
        Flips:
            - absolute y and snap_y (indices 1, 3): 53.3 - y reflects across midfield
            - relative y and dir_sin (indices 5, 7): negate since sin(360-θ) = -sin(θ)
        """
        aug_route = route.clone()
        aug_route[:length, [1,3]] = 53.3 - aug_route[:length, [1,3]] # flip y coords and snap_y
        aug_route[:length, [5,7]] = -aug_route[:length, [5,7]] # flip relative y coords and dir_sin 
        return aug_route
    
    def change_direction(self, route, length):
        """
        Mirrors the route along the long axis of the field (end zone to end zone).
        Equivalent to flipping which end zone the receiver is running toward.
        Valid because NFL tracking data standardizes play direction.

        Flips:
            - absolute x and snap_x (indices 0, 2): 120 - x reflects across midfield
            - relative x and dir_cos (indices 4, 8): negate since cos(180-θ) = -cos(θ)
        """
        aug_route = route.clone()
        aug_route[:length, [0,2]] = 120 - aug_route[:length, [0,2]] # flip x coords and snap_x
        aug_route[:length, [4,8]] = -aug_route[:length, [4,8]] # flip relative x coords and dir_cos
        return aug_route

    def inc_depth(self, route, length):
        """
        Increases the depth of the route by scaling each frame's displacement by a 
        random factor greater than 1. Simulates a receiver running a deeper version
        of the same route.

        Returns None if the augmented route goes out of bounds (0-120 yards on x axis,
        0-53.3 yards on y axis) — these samples are discarded as unrealistic.

        Also scales speed and distance traveled by the same factor since the receiver
        is covering more ground per timestep.
        """
        factor = random.choice([1.05, 1.1, 1.15])
        aug_route = route.clone()
        for frame in range(1, length):
            # scale displacement from previous frame and build on previous augmented position
            aug_route[frame, [0,1,4,5]] = ((route[frame, [0,1,4,5]] - route[frame-1, [0,1,4,5]]) * factor) + aug_route[frame-1, [0,1,4,5]]
            
			# discard if route goes out of bounds
            if not ((0 < aug_route[frame, 0] < 120) and (0 < aug_route[frame, 1] < 53.3)):
                return None
            
        # scale speed and distance by same factor
        aug_route[:length, [6,9]] *= factor
        return aug_route
    
    def dec_depth(self, route, length):
        """
        Decreases the depth of the route by scaling each frame's displacement by a
        random factor less than 1. Simulates a receiver running a shallower version
        of the same route.

        Returns None if the augmented route goes out of bounds.

        Also scales speed and distance traveled by the same factor.
        """
        factor = random.choice([0.85, 0.9, 0.95])
        aug_route = route.clone()
        for frame in range(1, length):
            # scale displacement from previous frame and build on previous augmented position
            aug_route[frame, [0,1,4,5]] = ((route[frame, [0,1,4,5]] - route[frame-1, [0,1,4,5]]) * factor) + aug_route[frame-1, [0,1,4,5]]
            
			# discard if route goes out of bounds
            if not ((0 < aug_route[frame, 0] < 120) and (0 < aug_route[frame, 1] < 53.3)):
                return None
        
		# scale speed and distance by same factor
        aug_route[:length, [6,9]] *= factor
        return aug_route
    
    def augment(self, X, y):
        """
        Augments the dataset by generating new route samples for underrepresented
        classes until all classes reach target_counts.

        Samples are cycled through in random order (without replacement until
        exhausted, then reshuffled) to ensure all real routes are seen before
        any is repeated.

        Augmentation strategy:
            - 25% chance: apply two augmentations in sequence
                - 50% chance: one spatial flip (change_side or change_direction)
                              followed by a depth change (inc_depth or dec_depth)
                - 50% chance: both spatial flips combined (change_side + change_direction)
            - 75% chance: apply a single random augmentation

        Args:
            X: tensor of shape (num_samples, max_frames, 11) — route sequences
            y: tensor of shape (num_samples,) — integer class labels

        Returns:
            X_aug: augmented route tensor
            y_aug: augmented label tensor
        """
        X_aug = X.clone()
        y_aug = y.clone()
        current_counts = torch.bincount(y)
        needed = self.target_counts - current_counts
        
        for idx, need in enumerate(needed):
            if need <= 0:
                continue # class already at or above target count
			
			# get all routes of this class
            route_idx = (y == idx)
            route_X = X[route_idx]
            route_y = y[route_idx]

			# shuffle indices — cycle through all routes before repeating any
            # code written with AI assistance
            perm = torch.randperm(len(route_y))
            perm_idx = 0
            while need > 0:
                # reshuffle once all routes have been used
                # code written with AI assistance
                if perm_idx >= len(perm):
                    perm = torch.randperm(len(route_y))
                    perm_idx = 0
                
                route = route_X[perm[perm_idx].item()]
                perm_idx += 1
                length = int(route[0,10].item()) # real route length from lengths column

                if random.random() < 0.25:
                    # apply two augmentations in sequence
                    if random.random() < 0.5:
                        # spatial flip + depth change
                        aug1 = random.choice([self.change_side, self.change_direction])
                        aug2 = random.choice([self.inc_depth, self.dec_depth])
                        aug_route = aug1(route, length)
                        aug_route = aug2(aug_route, length)
                    else:
                        # both spatial flips combined
                        aug_route = self.change_side(route, length)
                        aug_route = self.change_direction(aug_route, length)
                else:
                    # apply a single random augmentation
                    aug = random.choice([self.change_side, self.change_direction, self.inc_depth, self.dec_depth])
                    aug_route = aug(route, length)
                
				# only add if augmentation was valid (not None from out of bounds)
                if aug_route is not None:
                    X_aug = torch.cat([X_aug, aug_route.unsqueeze(0)], dim=0)
                    y_aug = torch.cat([y_aug, torch.tensor([idx])])
                    need -= 1
                    
        return X_aug, y_aug