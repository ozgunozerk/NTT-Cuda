# NTT-Cuda

An accelarated NTT, used in SEAL's Homomorphic Keygen, Encryption and Decryption operations.

This GPU implementation improve the performance of these three BFV operations by up to 141.95×, 105.17× and 90.13×, respectively, on Tesla v100 GPU compared to the highly-optimized SEAL library running on an Intel i9-7900X CPU.

Code is very complicated, please refer to the `Article.pdf` in the repo for details.

Readme may be improved later, but for now, try to understand from the inline-comments and the article.
