1. Implement the following three networks

The three following networks can all be implemented using different arguments and options from the jammy_flows

Länkar till en extern sida. library.

Hint: Install Jammy Flows with `pip install git+https://github.com/thoglu/jammy_flows.git --no-deps` to avoid any installation problems with additional dependencies you don't need for this exercise. 

1. Implement the following
    1. A Normalizing Flow consisting of a diagonal Gaussian, i.e., the free parameters of the normalizing flow are the mean and standard deviation of each of the three labels. This is equivalent to the previous exercise, but the mean and standard deviation is implemented by using an affine normalizing flow through the Jammy Flows package. 
    2. A Normalizing Flow consisting of a full 3-D Gaussian, i.e., the free parameters are the means of the three labels and the full covariance matrix between the three labels. 
    3. A Normalizing Flow consisting of several Gaussianization Flows plus one Affine Flow to predict arbitrary uncertainty PDFs. 

2. Quantify how well you were able to determine the uncertainties. 

3. Visualize a few predicted PDFs from the Gaussianization Flow. 

Implementing a Normalizing Flow model with Jammy Flows can be technically challenging. Therefore, we provide a template (B03train_normalizing_flow_template.py

Ladda ner B03train_normalizing_flow_template.py) that implements the model definition, the three flows you are supposed to use, and the loss function. If you use this code, please make sure that you understand it.

Remember always to include:

    A written summary (0.5–1 A4 page) covering (submitted either as PDF or directly as text):
        What you did and how
        What results you obtained
        What challenges you encountered and what could be improved
    A PDF (or similar format) with all result plots, each with a short explanation
    Your code, preferably as a link (e.g., GitHub, Google Colab, etc.) so we can view it easily.
