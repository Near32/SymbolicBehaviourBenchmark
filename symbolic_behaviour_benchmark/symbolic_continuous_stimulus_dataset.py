from typing import Dict, List

import sys
import random
import numpy as np 
import argparse 
import copy

eps = 1e-8


class SymbolicContinuousStimulusDataset:
    def __init__(
        self, 
        train=True, 
        transform=None, 
        sampling_strategy=None,
        split_strategy=None, 
        nbr_latents=10, 
        min_nbr_values_per_latent=2, 
        max_nbr_values_per_latent=10, 
        nbr_object_centric_samples=1,
        prototype=None) :
        '''
        :param sampling_stragey: str
            e.g.: 'component-focused-5shots'
        :param split_strategy: str 
            e.g.: 'divider-10-offset-0'
        '''
        self.nbr_latents = nbr_latents
        self.min_nbr_values_per_latent = min_nbr_values_per_latent
        self.max_nbr_values_per_latent = max_nbr_values_per_latent
        self.nbr_object_centric_samples = nbr_object_centric_samples
        
        self.test_set_divider = 2

        self.prototype = prototype
        
        self.train = train
        self.transform = transform
        self.sampling_strategy = sampling_strategy
        self.split_strategy = split_strategy

        self.reset()

    def reset_sampling(self):
        self.imgs = [np.zeros((64,64,3))]
        
        if self.split_strategy is not None:
            strategy = self.split_strategy.split('-')
            if 'divider' in self.split_strategy and 'offset' in self.split_strategy:
                self.divider = int(strategy[1])
                assert(self.divider>0)
                self.offset = int(strategy[-1])
                assert(self.offset>=0 and self.offset<self.divider)
            elif 'combinatorial' in self.split_strategy:
                self.counter_test_threshold = int(strategy[0][len('combinatorial'):])
                # (default: 2) Specifies the threshold on the number of latent dimensions
                # whose values match a test value. Below this threshold, samples are used in training.
                # A value of 1 implies a basic train/test split that tests generalization to out-of-distribution values.
                # A value of 2 implies a train/test split that tests generalization to out-of-distribution pairs of values...
                # It implies that test value are encountered but never when combined with another test value.
                # It is a way to test for binary compositional generalization from well known stand-alone test values.
                # A value of 3 tests for ternary compositional generalization from well-known:
                # - stand-alone test values, and
                # - binary compositions of test values.
                
                '''
                With regards to designing axises as primitives:
                
                It implies that all the values on this latent axis are treated as test values
                when combined with a test value on any other latent axis.
                
                N.B.: it is not possible to test for out-of-distribution values in that context...
                N.B.1: It is required that the number of primitive latent axis be one less than
                        the counter_test_thershold, at most.

                A number of fillers along this primitive latent axis can then be specified in front
                of the FP pattern...
                Among the effective indices, those with an ordinal lower or equal to the number of
                filler allowed will be part of the training set.
                '''

                nbr_primitives_and_tested = len([
                    k for k in self.latent_dims 
                    if self.latent_dims[k]['primitive'] or 'untested' not in self.latent_dims[k]
                ])
                #assert(nbr_primitives_and_tested==self.counter_test_threshold)
            else:
                raise NotImplementedError
        else:
            self.divider = 1
            self.offset = 0

        self.indices = []
        if self.prototype is not None:
            assert not(self.train)
            self.indices = [idx for idx in range(self.dataset_size) if idx not in self.prototype.indices]
            #print(f"Split Strategy: {self.split_strategy}")
            #print(f"Dataset Size: {len(self.indices)} out of {self.dataset_size} : {100*len(self.indices)/self.dataset_size}%.")

        elif self.split_strategy is None or 'divider' in self.split_strategy:
            for idx in range(self.dataset_size):
                if idx % self.divider == self.offset:
                    self.indices.append(idx)

            self.train_ratio = 0.8
            end = int(len(self.indices)*self.train_ratio)
            if self.train:
                self.indices = self.indices[:end]
            else:
                self.indices = self.indices[end:]
            #print(f"Split Strategy: {self.split_strategy} --> d {self.divider} / o {self.offset}")
            #print(f"Dataset Size: {len(self.indices)} out of {self.dataset_size}: {100*len(self.indices)/self.dataset_size}%.")
        elif 'combinatorial' in self.split_strategy:
            for idx in range(self.dataset_size):
                object_centric_sidx = idx//self.nbr_object_centric_samples
                coord = self.idx2coord(object_centric_sidx)
                latent_class = np.array(coord)
                
                effective_test_threshold = self.counter_test_threshold
                counter_test = {}
                skip_it = False
                filler_forced_training = False
                for dim_name, dim_dict in self.latent_dims.items():
                    dim_class = latent_class[dim_dict['position']]
                    quotient = (dim_class+1)//dim_dict['divider']
                    remainder = (dim_class+1)%dim_dict['divider']
                    if remainder!=dim_dict['remainder_use']:
                        skip_it = True

                    if dim_dict['primitive']:
                        ordinal = quotient
                        if ordinal > dim_dict['nbr_fillers']:
                            effective_test_threshold -= 1

                    if 'test_set_divider' in dim_dict and quotient%dim_dict['test_set_divider']==0:
                        counter_test[dim_name] = 1
                    elif 'test_set_size_sample_from_end' in dim_dict:
                        max_quotient = dim_dict['size']//dim_dict['divider']
                        if quotient > max_quotient-dim_dict['test_set_size_sample_from_end']:
                            counter_test[dim_name] = 1
                    elif 'test_set_size_sample_from_start' in dim_dict:
                        max_quotient = dim_dict['size']//dim_dict['divider']
                        if quotient <= dim_dict['test_set_size_sample_from_start']:
                            counter_test[dim_name] = 1

                    if dim_name in counter_test:
                        self.test_latents_mask[idx, dim_dict['position']] = 1
                        
                if skip_it: continue


                if self.train:
                    if len(counter_test) >= effective_test_threshold:#self.counter_test_threshold:
                        continue
                    else:
                        self.indices.append(idx)
                else:
                    if len(counter_test) >= effective_test_threshold:#self.counter_test_threshold:
                        self.indices.append(idx)
                    else:
                        continue

            #print(f"Split Strategy: {self.split_strategy}")
            #print(f"Dataset Size: {len(self.indices)} out of {self.dataset_size} : {100*len(self.indices)/self.dataset_size}%.")
        else:
            raise NotImplementedError            

        """
        self.targets contains stimulus-centric indices as keys:
        """
        self.indices_aligned_targets = self.targets[self.indices]
        
        """
        self.trueidx2idx expects stimulus-centric indices both as keys and values,
        as self.indices.
        The values represent the indices, as seen from the outside.
        The keys represent the indices from the original dataset, inside-view...
        This mapping goes from original indices to ordered indices as seen from the outside.
        """
        self.trueidx2idx = dict(zip(self.indices,range(len(self.indices))))

        self.sampling_indices = None
        if self.sampling_strategy is not None\
        and 'component-focused' in self.sampling_strategy:
            """
            Expects: 'component-focused-Xshots'
            where X is an integer value representing
            the number of times each component value must be seen.
            """
            assert 'shot' in self.sampling_strategy
            nbr_shots = int(self.sampling_strategy.split('-')[-1].split('shot')[0])
            """
            Starting with uniform distribution over all values, on each latent dim,
            the weights of the distribution are initialised at the number of shots,
            plus one for regularisation purposes.
            We aim to sample indices using a building loop.
            This building loop goes through all latent dimensions 
            and sample a value according to the weights.
            Upon sampling a component value, the corresponding weight is then decreased by one.
            This ensures that all values, that needs to be sampled in order to be seen the correct
            number of shots, retain high likelihood of being sampled.
            """
            step_size = 100.0
            self.per_latent_value_weights = {
                lidx: [(nbr_shots+1)*step_size for vidx in range(ld['size'])]
                for lidx, ld in self.latent_dims.items()
            }
            
            """
            It may be possible that the structure does not accomodate
            the number of shots for each component.
            Thus, we need to allow for replacement when sampling coords below,
            at training time, or simply sampling everything, at test time.
            """
            allow_replacement = False
            nbr_trials_threshold = 10
            nbr_trials_before_accepting_replacement = 5
            need_sampling = True

            """
            self.indices is expecting stimulus-centric indices both as keys and values.
            The method self.idx2coord is expecting object-centric indices.
            Thus, many coords in valid_coords are the same, but are indexed via stimulus-centric indices...
            valid_coords[ from-the-outside-scidx ] 
            """
            valid_coords = np.stack(
                [self.idx2coord(trueidx//self.nbr_object_centric_samples) for trueidx in self.indices],
                axis=0,
            )
            # very rough estimation of the minimum number of samples needed:
            # ... assuming incorrectly that when sampling for one value, it does not already
            # gives us samples for some other values on other latent dimensions...
            min_nbr_samples = nbr_shots*sum([ld['size'] for ld in self.latent_dims.values()])
            if min_nbr_samples > len(self.indices):
                if self.train:
                    allow_replacement = True
                else:
                    self.sampling_indices = list(range(len(self.indices)))
                    need_sampling = False
            
            if need_sampling:
                self.sampling_indices = []
                
                done = False
                sampled_coord = []
                nbr_trials = 0
                while not done:
                    coord = np.zeros(self.nbr_latents)
                    for lidx, vws in self.per_latent_value_weights.items():
                        norm = sum(vws)
                        probs = [vw/norm for vw in vws]
                        sampled_vidx = np.random.choice(
                            a=np.arange(len(vws)),
                            size=1,
                            p=probs,
                        )
                    
                        coord[lidx] = sampled_vidx 
                
                    # Check that we have not already sampled it:
                    same = len(sampled_coord) and any([all(coord==c) for c in sampled_coord])
                    if same \
                    and not allow_replacement:
                        nbr_trials += 1
                        if nbr_trials > nbr_trials_threshold:
                            allow_replacement = True
                            nbr_trials = 0
                        continue
                    elif same \
                    and allow_replacement:
                        if nbr_trials < nbr_trials_before_accepting_replacement:
                            nbr_trials += 1
                            continue
                        else:
                            nbr_trials = 0
                    
                    # Convert to sample's trueidx:
                    sampled_obj_centric_trueidx = self.coord2idx(coord)
                    # sampled_obj_centric_trueidx is an object-centric index...
                    # BEWARE: random's randint gives values within low and high, both incluse!!!
                    # on the contrary to numpy's random.randint which is low(inclusive) and high(exclusive).
                    sampled_trueidx = sampled_obj_centric_trueidx*self.nbr_object_centric_samples + random.randint(0, self.nbr_object_centric_samples-1)
                    # Record for sampling, iff valid trueidx:
                    # Otherwise, we need to sample again...
                    if sampled_trueidx in self.trueidx2idx:
                        """
                        self.sampling_indices should expect stimulus-centric indices
                        both as keys and values.
                        """
                        self.sampling_indices.append(self.trueidx2idx[sampled_trueidx])
                    else:
                        # let us sample the coord among the valid ones
                        # that resembles the most this one:
                        coord_l2_dists = np.square(10.0*(valid_coords-coord))
                        coord_w_weights = copy.deepcopy(coord_l2_dists)
                        for lidx, vws in self.per_latent_value_weights.items():
                            inf_norm = max(vws)
                            coord_w_weights[:, lidx] = inf_norm/(coord_w_weights[:, lidx]+1.0e-3)
                        coord_w_weights = coord_w_weights.sum(axis=-1)
                        coord_weights = 1.0/(coord_l2_dists.sum(axis=-1)+1.0e-3)
                        norm = coord_weights.sum()
                        probs = [cw/norm for cw in coord_weights]
                        # inversely proportional to the distance:
                        replacement_idx = np.random.choice(
                            a=len(probs),
                            size=1,
                            p=probs,
                        ).item()
                        #replacement_trueidx = self.indices[replacement_idx]
                        #self.sampling_indices.append(self.trueidx2idx[replacement_trueidx])
                        """
                        self.sampling_indices should expect stimulus-centric indices.
                        As valid_coords is made up of object-centric coords, and indexed with
                        stimulus-centric indices, replacement_idx is chosen among stimulus-centric
                        indices, giving equal likelihood to same ranges of object-centric indices...
                        """
                        self.sampling_indices.append(replacement_idx)
                        coord = valid_coords[replacement_idx]
                        
                    # Bookkeeping:
                    sampled_coord.append(coord)

                    for lidx, sampled_vidx in enumerate(coord):
                        # Decreasing likelihood of this value for next runs:
                        vw = self.per_latent_value_weights[lidx]
                        
                        # ...but keep a minimum probability of sampling.
                        # Otherwise, we might find ourselves in the situation
                        # where we still need to sample values from one latent
                        # dimension while all the other latent dimensions have
                        # their weight distributions at 0...
                        
                        self.per_latent_value_weights[lidx][int(sampled_vidx)] = max(
                            step_size,
                            vw[int(sampled_vidx)]-step_size,
                        )
                    
                     
                    # Are we done?
                    # i.e. are all the weights distr equal to their final value: step_size?
                    done = True
                    for lidx, vw in self.per_latent_value_weights.items():
                        if sum(vw)/len(vw) > step_size:
                            done = False
                            break
            else:
                pass
                #print(f"WARNING: Sampling was not performed due to not enough available stimulus.")

            pass
            #print(f"Sampling indices length: {len(self.sampling_indices)} out of {len(self.indices)} : {len(self.sampling_indices)/len(self.indices)*100} %.")
            #print(self.sampling_indices)
        #print('Dataset loaded : OK.')

    def reset(self):
        global eps 
        
        if self.prototype is None:
            self.latent_dims = {}
            self.latent_sizes = []
            self.dataset_size = 1
            for l_idx in range(self.nbr_latents):
                l_size = np.random.randint(low=self.min_nbr_values_per_latent, high=self.max_nbr_values_per_latent+1)
                self.dataset_size *= l_size
                self.latent_sizes.append(l_size)
                self.latent_dims[l_idx] = {'size': l_size}
                
                self.latent_dims[l_idx]['value_section_size'] = 2.0/l_size
                self.latent_dims[l_idx]['max_sigma'] = self.latent_dims[l_idx]['value_section_size']/6
                self.latent_dims[l_idx]['min_sigma'] = self.latent_dims[l_idx]['value_section_size']/12
                self.latent_dims[l_idx]['sections'] = {}
                for s_idx in range(l_size):
                    s_d = {}
                    s_d['section_offset'] = -1+s_idx*self.latent_dims[l_idx]['value_section_size']
                    s_d['sigma'] = np.random.uniform(
                        low=self.latent_dims[l_idx]['min_sigma']+eps,
                        high=self.latent_dims[l_idx]['max_sigma']-eps,
                    )
                    s_d['safe_section_size'] = self.latent_dims[l_idx]['value_section_size'] - 6*s_d['sigma']
                    s_d['safe_section_mean_offset'] = 3*s_d['sigma']
                    s_d['mean_ratio'] = np.random.uniform(low=0,high=1.0)
                    s_d['mean'] = s_d['section_offset'] + s_d['safe_section_mean_offset'] + s_d['mean_ratio'] * s_d['safe_section_size']
                    
                    self.latent_dims[l_idx]['sections'][s_idx] = s_d

                self.latent_dims[l_idx]['nbr_fillers'] = 0
                self.latent_dims[l_idx]['primitive'] = False
                self.latent_dims[l_idx]['position'] = l_idx
                self.latent_dims[l_idx]['remainder_use'] = 0
                self.latent_dims[l_idx]['divider'] = 1 # no need to divide as it is fully parameterized
                self.latent_dims[l_idx]['test_set_divider'] = self.test_set_divider

            self.dataset_size *= self.nbr_object_centric_samples
            self.generate_object_centric_samples()

            self.latent_strides = [1]
            dims = [ld['size'] for ld in self.latent_dims.values()]
            for idx in range(self.nbr_latents):
                self.latent_strides.append(np.prod(dims[-idx-1:]))
            self.latent_strides = list(reversed(self.latent_strides[:-1]))
            
            self.test_latents_mask = np.zeros((self.dataset_size, self.nbr_latents))
        else:
            self.latent_dims = self.prototype.latent_dims
            self.latent_sizes = self.prototype.latent_sizes
            self.dataset_size = self.prototype.dataset_size
            self.latent_strides = self.prototype.latent_strides
            self.test_latents_mask = self.prototype.test_latents_mask

        self.targets = np.zeros(self.dataset_size)
        """
        self.targets contains stimulus-centric indices,
        and the targets/values correspond to the object-centric indices.
        """
        for idx in range(self.dataset_size):
            self.targets[idx] = idx//self.nbr_object_centric_samples
       
        self.reset_sampling()
        self.reset_OC_classes()
    
    def get_OC_classes(self):
        if not hasattr(self, 'OC_classes'):
            self.reset_OC_classes()
        return self.OC_classes

    def reset_OC_classes(self):
        self.OC_classes = {}
        for idx in range(self.dataset_size):
            cl = self.targets[idx]
            if cl not in self.OC_classes: self.OC_classes[cl] = []
            self.OC_classes[cl].append(idx)

    def generate_object_centric_samples(self):
        """
        """
        for lidx in range(self.nbr_latents):
            for lvalue in range(self.latent_dims[lidx]['size']):
                oc_samples = []
                for oc_sidx in range(self.nbr_object_centric_samples):
                    lvalue_sample = np.random.normal(
                        loc=self.latent_dims[lidx]['sections'][lvalue]['mean'],
                        scale=self.latent_dims[lidx]['sections'][lvalue]['sigma'],
                    )
                    oc_samples.append(float(lvalue_sample))
                self.latent_dims[lidx]['sections'][lvalue]['object_centric_samples'] = oc_samples

    def generate_object_centric_observations(
        self, 
        latent_class:np.ndarray,
        object_centric_sample_idx:int=None):
        """
        :param latent_class: Numpy.ndarray of shape (batch_size, self.nbr_latents).

        :return observations: Numpy.ndarray of shape (batch_size, self.nbr_latents) with
            values on each dimension sampled from the corresponding value's (gaussian) 
            distribution.
        """
        batch_size = latent_class.shape[0]
        if object_centric_sample_idx is None:
            object_centric_sample_idx = np.random.randint(low=0,high=self.nbr_object_centric_samples)

        observations = np.zeros((batch_size, self.nbr_latents))
        for bidx in range(batch_size):
            for lidx in range(self.nbr_latents):
                lvalue = latent_class[bidx,lidx]
                lvalue_sample = self.latent_dims[lidx]['sections'][lvalue]['object_centric_samples'][object_centric_sample_idx]
                observations[bidx,lidx] = float(lvalue_sample)

        return observations

  
    def generate_observations(self, latent_class, sample=True):
        """
        :arg latent_class: Numpy.ndarray of shape (batch_size, self.nbr_latents).
        :arg sample: Bool, if `True`, then values are sampled from each distribution.
            Otherwise, the mean value is used.
        :return observations: Numpy.ndarray of shape (batch_size, self.nbr_latents) with
            values on each dimension sampled from the corresponding value's (gaussian) 
            distribution.
        """
        batch_size = latent_class.shape[0]

        observations = np.zeros((batch_size, self.nbr_latents))
        for bidx in range(batch_size):
            for lidx in range(self.nbr_latents):
                lvalue = latent_class[bidx,lidx]
                if sample:
                    lvalue_sample = np.random.normal(
                        loc=self.latent_dims[lidx]['sections'][lvalue]['mean'],
                        scale=self.latent_dims[lidx]['sections'][lvalue]['sigma'],
                    )
                else:
                    lvalue_sample = self.latent_dims[lidx]['sections'][lvalue]['mean']
                observations[bidx,lidx] = float(lvalue_sample)

        return observations

    def coord2idx(self, coord):
        """
        WARNING: the object-centrism is not taken into account here.
        I.e. in order to obtain a stimulus from the :return idx: value,
        it is necessary to:
            - retrieve the latentclass from this index (using getlatentclass),
            - and call the method generate_object_centric_observations with the
                retrieved latentclass as input.
        The object-centric observations is sampled randomly in the 
        generate_object_centric_observations method.

        :arg coord: List of self.nbr_latents elements.        
        
        :return idx: Integer, corresponding stimulus-centric index.
        """
        idx = 0
        for stride, mult in zip(self.latent_strides,coord):
            idx += stride*mult
        return idx

    def idx2coord(self, idx):
        """
        WARNING: the object-centrism MUST be taken into account
        before calling this function.
        E.g:
            - idx2coord( self.targets[self.indices[self.sampling_indices[i]]] )
            - idx2coord( self.indices_aligned_targets[self.sampling_indices[i]] )

        :arg idx: Integer representing an object-centric index,
                    i.e. must be contained within [0, self.dataset_size/self.nbr_object_centric_samples].

        :return coord: List of self.nbr_latents elements corresponding the entry of :arg idx:.
        """
        coord = []
        remainder = idx
        for lidx in range(self.nbr_latents):
            coord.append(remainder//self.latent_strides[lidx])
            remainder = remainder % self.latent_strides[lidx]
        return coord 
    
    def __len__(self) -> int:
        if self.sampling_indices is not None:
            return len(self.sampling_indices)
            #return self.nbr_object_centric_samples*len(self.sampling_indices)

        return len(self.indices)

    def getclass(self, idx=None, sidx=None, stimulus_centric=False):
        """
        :param idx: Integer representing the stimulus index.
                    If self.sampling_indices is not None, i.e. if using
                    'component-focused'-based sampling strategy,
                    then it is assumed that self.sampling_indices contains
                    stimulus-centric indices as keys and values, and 
                    self.targets containing stimulus-centric indices as keys.
        :param sidx: Integer representing the object-centric stimulus index,
                    i.e. divided by self.nbr_object_centric_samples already.
        :param stimulus_centric: Boolean assert whether sidx is an Integer representing 
                        a stimulus-centric index, i.e. not aligned with self.indices, 
                        but with self.targets.
        """
        assert idx is not None or sidx is not None
        if idx is not None\
        and self.sampling_indices is not None:
            #idx = self.sampling_indices[idx//self.nbr_object_centric_samples]
            idx = self.sampling_indices[idx]
        elif sidx is not None:
            # sampling idx is provided already:
            idx = sidx

        if stimulus_centric:
            target = self.targets[idx]
        else:
            idx = idx%len(self.indices)
            #target = self.targets[idx]
            target = self.indices_aligned_targets[idx]
        return target

    def getobjectcentricsampleidx(self, idx=None, scidx=None):
        """
        :param idx: Integer in stimulus-centric fashion, and self.indices aligned.
        :param scidx: Integer in stimulus-centric fashion, but not aligned with self.indices.
        :return:
            - object_centric_sid: Integer within [0, nbr_object_centric_samples-1] which
            identifies the i-th stimulus in the object-centric class of the stimulus indexed
            with :param idx:.
        """
        assert idx is not None or scidx is not None

        if idx is not None:
            idx = idx%len(self.indices)
            trueidx = self.indices[idx]
        else:
            trueidx = scidx

        object_centric_sidx = trueidx % self.nbr_object_centric_samples
        return object_centric_sidx

    def getstimuluscentricsampleidx(self, idx):
        """
        :param idx: Integer of a stimulus, as seen from the outside, i.e. not necessarily
                    stimulus-centric yet (with respect to sampling indices and indices remapping...).
        :return:
            - stimulus_centric_sid: Integer within [0, self.dataset_size] which
            identifies in a stimulus-centric fashion with respect to self.targets keys.
        """
        if self.sampling_indices is not None:
            idx = self.sampling_indices[idx]
        idx = idx%len(self.indices)
        truesidx = self.indices[idx]
        return truesidx

    def getlatentvalue(self, idx, stimulus_centric=False):
        if not stimulus_centric:
            idx = idx%len(self.indices)
            trueidx = self.indices[idx]
        else:
            trueidx = idx
        object_centric_sidx = trueidx//self.nbr_object_centric_samples
        coord = self.idx2coord(object_centric_sidx)
        latent_class = np.array(coord).reshape((1,-1))
        latent_value = self.generate_observations(latent_class, sample=False)
        return latent_value

    def getlatentclass(self, idx, stimulus_centric=False):
        if stimulus_centric:
            trueidx = idx
        else:
            idx = idx%len(self.indices)
            trueidx = self.indices[idx]
        object_centric_sidx = trueidx//self.nbr_object_centric_samples
        coord = self.idx2coord(object_centric_sidx)
        latent_class = np.array(coord)
        return latent_class

    def getlatentonehot(self, idx, stimulus_centric=False):
        # object-centrism is taken into account in getlatentclass fn:
        latent_class = self.getlatentclass(idx, stimulus_centric=stimulus_centric)
        latent_one_hot_encoded = np.zeros(sum(self.latent_sizes))
        startidx = 0
        for lsize, lvalue in zip(self.latent_sizes, latent_class):
            latent_one_hot_encoded[startidx+lvalue] = 1
            startidx += lsize 
        return latent_one_hot_encoded

    def gettestlatentmask(self, idx, stimulus_centric=False):
        if stimulus_centric:
            trueidx = idx
        else:
            idx = idx%len(self.indices)
            trueidx = self.indices[idx]
        test_latents_mask = self.test_latents_mask[trueidx]
        return test_latents_mask

    def sample_factors(self, num, random_state):
        """
        Sample a batch of factors Y.
        """
        #return random_state.randint(low=0, high=self.nbr_values_per_latent, size=(num, self.nbr_latents))
        # It turns out the random state is not really being updated apparently.
        # Therefore it was always sampling the same values...
        random_indices = np.random.randint(low=0, high=self.dataset_size, size=(num,))
        return np.stack([self.getlatentclass(ridx) for ridx in random_indices], axis=0)
        
    def sample_observations_from_factors(self, factors, random_state):
        """
        Sample a batch of observations X given a batch of factors Y.
        """
        return self.generate_object_centric_observations(factors)

    def sample_latents_values_from_factors(self, factors, random_state):
        """
        Sample a batch of observations X given a batch of factors Y.
        """
        return self.generate_observations(factors, sample=False)

    def sample_latents_ohe_from_factors(self, factors, random_state):
        """
        Sample a batch of observations X given a batch of factors Y.
        """
        batch_size = factors.shape[0]
        ohe = np.zeros((batch_size, sum(self.latent_sizes)))
        for bidx in range(batch_size):
            idx = self.coord2idx(factors[idx])
            ohe[bidx] = self.getlatentonehot(idx)
        return ohe

    def sample(self, num, random_state):
        """
        Sample a batch of factors Y and observations X.
        """
        factors = self.sample_factors(num, random_state)
        return factors, self.sample_observations_from_factors(factors, random_state)

    def __getitem__(self, idx:int=None, sidx:int=None) -> Dict[str,np.ndarray]:
        """
        :param idx: Integer index.
        :param sidx: Integer index, on a stimulus-centric basis, i.e. overriding the sampling_indices remapping.

        :returns:
            sampled_d: Dict of:
                - `"experiences"`: Tensor of the sampled experiences.
                - `"exp_labels"`: List[int] consisting of the indices of the label to which the experiences belong.
                - `"exp_latents"`: Tensor representation of the latent of the experience in one-hot-encoded vector form.
                - `"exp_latents_values"`: Tensor representation of the latent of the experience in value form.
                - `"exp_latents_one_hot_encoded"`: Tensor representation of the latent of the experience in one-hot-encoded class form.
                - `"exp_test_latent_mask"`: Tensor that highlights the presence of test values, if any on each latent axis.
        """
        assert idx is not None or sidx is not None
        object_centric_sample_idx = None

        if idx is not None \
        and self.sampling_indices is not None:
            """
            PREVIOUSLY:
            self.sampling_indices is assuming object-centric indices as keys,
            and stimulus-centric indices as values, stimulus-centric indices 
            are expected by self.getlatentclass method.
            NOW: self.sampling_indices uses stimulus-centric indices as keys 
            and values.
            """
            #idx = self.sampling_indices[idx//self.nbr_object_centric_samples]
            idx = self.sampling_indices[idx]
        
        if sidx is not None:
            idx = sidx
            # since it is a stimulus-centric index, we can extract the class
            # it belongs to from its value:
            # using idx as entry since it is an index aligned with self.indices
            object_centric_sample_idx = self.getobjectcentricsampleidx(idx=idx)

        latent_class = self.getlatentclass(idx)
        # Does this latent class vectors contains the object-centric label?
        # No, but it is identified in object_centric_sample_idx:
        stimulus = self.generate_object_centric_observations(
            latent_class.reshape((1,-1)),
            object_centric_sample_idx=object_centric_sample_idx,
        )
       
        # PREVIOUSLY: regularised getclass expectations?
        # now expecting stimulus-centric values, but still
        # acknowledging that it is a sampled index...
        if sidx is not None \
        and self.sampling_indices is not None:
            target = self.getclass(sidx=idx)
        elif self.sampling_indices is not None:
            target = self.getclass(sidx=idx)
        else:
            target = self.getclass(idx)
        
        latent_value = self.getlatentvalue(idx)
        latent_one_hot_encoded = self.getlatentonehot(idx)
        test_latents_mask = self.gettestlatentmask(idx)

        if self.transform is not None:
            stimulus = self.transform(stimulus)
        
        # POSDIS utterance as RNN STATES
        rnn_states = latent_class+1
         
        sampled_d = {
            "experiences":stimulus, 
            #"experiences":latent_class[np.newaxis,...], #stimulus, 
            "exp_labels":target, 
            "exp_latents":latent_class, 
            "exp_latents_values":latent_value,
            "exp_latents_one_hot_encoded":latent_one_hot_encoded,
            "exp_test_latents_masks":test_latents_mask,
            ##
            "rnn_states":rnn_states,
        }

        return sampled_d

    def getstimuluscentricstimulus(self, sidx:int) -> Dict[str,np.ndarray]:
        """
        :param sidx: Integer index, on a stimulus-centric basis, i.e. overriding the sampling_indices remapping.

        :returns:
            sampled_d: Dict of:
                - `"experiences"`: Tensor of the sampled experiences.
                - `"exp_labels"`: List[int] consisting of the indices of the label to which the experiences belong.
                - `"exp_latents"`: Tensor representation of the latent of the experience in one-hot-encoded vector form.
                - `"exp_latents_values"`: Tensor representation of the latent of the experience in value form.
                - `"exp_latents_one_hot_encoded"`: Tensor representation of the latent of the experience in one-hot-encoded class form.
                - `"exp_test_latent_mask"`: Tensor that highlights the presence of test values, if any on each latent axis.
        """
        idx = sidx
        # since it is a stimulus-centric index, we can extract the class
        # it belongs to from its value:
        # using scidx as entry since it is an index not aligned with self.indices, but directly with self.targets
        object_centric_sample_idx = self.getobjectcentricsampleidx(scidx=idx)

        latent_class = self.getlatentclass(idx, stimulus_centric=True)
        # Does this latent class vectors contains the object-centric label?
        # No, but it is identified in object_centric_sample_idx:
        stimulus = self.generate_object_centric_observations(
            latent_class.reshape((1,-1)),
            object_centric_sample_idx=object_centric_sample_idx,
        )
       
        # PREVIOUSLY: regularised getclass expectations?
        # now expecting stimulus-centric values, but still
        # acknowledging that it is a sampled index...
        target = self.getclass(sidx=idx, stimulus_centric=True)
        
        latent_value = self.getlatentvalue(idx, stimulus_centric=True)
        latent_one_hot_encoded = self.getlatentonehot(idx, stimulus_centric=True)
        test_latents_mask = self.gettestlatentmask(idx, stimulus_centric=True)

        if self.transform is not None:
            stimulus = self.transform(stimulus)
        
        sampled_d = {
            "experiences":stimulus.astype(np.float32), 
            "exp_labels":target.astype(int), 
            "exp_latents":latent_class.astype(np.float32), 
            "exp_latents_values":latent_value.astype(np.float32),
            "exp_latents_one_hot_encoded":latent_one_hot_encoded.astype(np.float32),
            "exp_test_latents_masks":test_latents_mask.astype(np.float32),
        }

        return sampled_d
