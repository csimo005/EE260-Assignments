# EE260-Assignments
A GitHub repository for my EE-260 assignments. Since we are going to be implementing a lot of stuff from scratch, I am
going to try to implement the assignments as my own deep learning framework as a gag. My end goal is for the from
scratch implementation to look as similar to pytorch as possible. Hopefully if will be an educational experience :p.

## To Do:
- [x] Fix module importing errors
- [x] Write Import Tests
- [x] Write MLP Tests
- [x] Defualt backwards parameter of 1, i.e. loss.backward()
- [ ] Implement tensor reshaping
    - [ ] Forward
    - [ ] Backward
- [x] Logistic loss
    - [x] Forward
    - [x] Backward
- [ ] Add option for regular labels in MNIST

## Known Bug List:
- MNIST pads batch size that don't perfectly divide len
- Logistic Loss causes nan, probably a divide by zero error
