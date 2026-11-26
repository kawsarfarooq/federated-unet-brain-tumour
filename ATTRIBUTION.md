# Attribution and third-party notices

## Methods and source context

This implementation follows published U-Net and FedAvg methods. The motivating study and its repository are acknowledged in [REFERENCES.md](REFERENCES.md). The implementation does not import or bundle that repository's source. The baseline reproduction records under results/reproduction use a separate source-faithful protocol.

## Source license notes

No project-wide LICENSE file was found in the inspected motivating repository at commit 9255825669277f0c64d80f970cf40b8de83d3901. Its unet_model/model.py contains Intel copyright and Apache-2.0 boilerplate alongside an EPL-2.0 SPDX identifier. These conflicting notices are not resolved here. This repository does not grant permission to relicense that source.

## Data and dependencies

Medical Segmentation Decathlon data is distributed under CC BY-SA 4.0. No medical volumes are bundled here; the illustrative training-data panel is attributed to the dataset in REFERENCES.md and carries the dataset's CC BY-SA 4.0 terms. TensorFlow, tf-keras, NumPy, NiBabel, Matplotlib, and pytest retain their respective licenses.

No repository-wide distribution license has been assigned to the implementation. Source availability does not itself grant a general reuse license. Existing third-party rights and notices remain applicable.
