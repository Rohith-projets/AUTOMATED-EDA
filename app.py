import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import klib as krishna
import numpy as np
import random
from matplotlib import *
from streamlit_option_menu import option_menu
import chardet




class Statistics:
    def __init__(self, dataset):
        self.dataset = dataset

    def basic_details(self):
        st.subheader("Basic Details")

        st.text(f"Number of Rows: {self.dataset.shape[0]}")
        st.text(f"Number of Columns: {self.dataset.shape[1]}")
        st.text(f"Number of Missing Values: {self.dataset.isnull().sum().sum()}")
        st.text(f"Size of the Dataset: {self.dataset.size}")
        st.text(f"Column Names: {self.dataset.columns.tolist()}")

        st.subheader("First 5 Rows:")
        st.dataframe(self.dataset.head())

        st.subheader("Last 10 Rows:")
        st.dataframe(self.dataset.tail(10))

        st.subheader("Random Sample (20% of Data):")
        st.dataframe(self.dataset.sample(frac=0.2))

    def secondary_information(self):
        st.subheader("Secondary Information")

        st.text("Column Data Types:")
        st.dataframe(self.dataset.dtypes)

        st.text("Memory Usage:")
        memory_usage_df = pd.DataFrame(self.dataset.memory_usage(deep=True), columns=['Memory Usage (bytes)'])
        st.dataframe(memory_usage_df)

        # Display numerical data types if they exist
        numerical_data = self.dataset.select_dtypes(include=['number','int32','int64','float32','float64'])
        if not numerical_data.empty:
            st.subheader("Numerical Data Columns:")
            st.dataframe(numerical_data)

        # Display categorical and time series data if they exist
        categorical_data = self.dataset.select_dtypes(include=['category', 'object','string'])
        time_series_data = self.dataset.select_dtypes(include=['datetime'])

        if not categorical_data.empty:
            st.subheader("Categorical Data Columns:")
            st.dataframe(categorical_data)
        
        if not time_series_data.empty:
            st.subheader("Time Series Data Columns:")
            st.dataframe(time_series_data)

    def statistics_1(self):
        st.subheader("Statistics - 1")

        # Display basic statistical summary
        st.text("Statistical Summary (describe):")
        st.dataframe(self.dataset.describe())

        # Display DataFrame information
        st.text("DataFrame Info:")
        st.write(self.dataset.info())
        
        
    def statistics_2(self):
        st.subheader("Statistics - 2")
        if True:
            st.text("Mean Values (by Columns):")
            mean_df = self.dataset.mean(numeric_only=True,skipna=True)  # Calculate mean across columns
            st.dataframe(mean_df)
            plt.figure(figsize=(10, 4))
            plt.bar(mean_df.index,mean_df.values)
            plt.plot(mean_df.index.tolist(),mean_df.values.tolist(),color="red",marker="o",mec="black",mfc="blue")
            plt.xticks(rotation=45, ha='right')
            plt.title("Mean Values (by Columns)")
            st.pyplot(plt)  # Display the figure

            st.text("Median Values (by Columns):")
            median_df = self.dataset.median(numeric_only=True,skipna=True)  # Calculate median across columns
            st.dataframe(median_df)
            plt.figure(figsize=(10, 4))
            plt.bar(mean_df.index,mean_df.values)
            plt.plot(mean_df.index.tolist(),mean_df.values.tolist(),color="red",marker="o",mec="black",mfc="blue")
            plt.xticks(rotation=45, ha='right')
            plt.title("Median Values (by Columns)")
            st.pyplot(plt.gcf())

            st.text("Mode Values (by Columns):")
            mode_df = self.dataset.mode(numeric_only=False)  # Calculate mode across columns
            st.dataframe(mode_df)
            plt.figure(figsize=(10, 4))
            sns.countplot(x=mode_df.index)
            plt.xticks(rotation=45, ha='right')
            plt.title("Mode Values (by Columns)")
            st.pyplot(plt.gcf())

            st.text("Correlation Matrix (by Columns):")
            corr_df = self.dataset.corr(numeric_only=True)  # Calculate correlation matrix across columns
            st.dataframe(corr_df)
            plt.figure(figsize=(10, 4))
            sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm')
            plt.title("Correlation Matrix (by Columns)")
            st.pyplot(plt.gcf())

            st.text("Covariance Matrix (by Columns):")
            cov_df = self.dataset.cov(numeric_only=True)  # Calculate covariance matrix across columns
            st.dataframe(cov_df)
            plt.figure(figsize=(10, 4))
            sns.heatmap(cov_df, annot=True, fmt='.2f', cmap='coolwarm')
            plt.title("Covariance Matrix (by Columns)")
            st.pyplot(plt.gcf())

            st.text("Variance (by Columns):")
            var_df = self.dataset.var(numeric_only=True,skipna=True)  # Calculate variance across columns
            st.dataframe(var_df)
            plt.figure(figsize=(10, 4))
            plt.bar(var_df.index, var_df.values)
            plt.plot(mean_df.index.tolist(),mean_df.values.tolist(),color="red",marker="o",mec="black",mfc="blue")
            plt.xticks(rotation=45, ha='right')
            plt.title("Variance (by Columns)")
            st.pyplot(plt)

            st.text("Standard Deviation (by Columns):")
            std_df = self.dataset.std(numeric_only=True,skipna=True)  # Calculate standard deviation across columns
            st.dataframe(std_df)
            plt.figure(figsize=(10, 4))
            plt.bar(std_df.index, std_df.values)
            plt.plot(std_df.index.tolist(),mean_df.values.tolist(),color="red",marker="o",mec="black",mfc="blue")
            plt.xticks(rotation=45, ha='right')
            plt.title("Standard Deviation (by Columns)")
            st.pyplot(plt)

            st.text("Standard Error of Mean (by Columns):")
            sem_df = self.dataset.sem(numeric_only=True,skipna=True)  # Calculate SEM across columns
            st.dataframe(sem_df)
            plt.figure(figsize=(10, 4))
            plt.bar(sem_df.index, sem_df.values)
            plt.plot(list(sem_df.index), sem_df.values.tolist(),color="red",marker="o",mec="black",mfc="blue")
            plt.xticks(rotation=45, ha='right')
            plt.title("Standard Error of Mean (by Columns)")
            st.pyplot(plt.gcf())

            st.text("Skewness (by Columns):")
            skew_df = self.dataset.skew(numeric_only=True,skipna=True)  # Calculate skewness across columns
            st.dataframe(skew_df)
            plt.figure(figsize=(10, 4))
            plt.bar(skew_df.index, skew_df.values)
            plt.plot(skew_df.index.tolist(), skew_df.values.tolist(),color="red",marker="o",mec="black",mfc="blue")
            plt.xticks(rotation=45, ha='right')
            plt.title("Skewness (by Columns)")
            st.pyplot(plt.gcf())

            st.text("Kurtosis (by Columns):")
            kurt_df = self.dataset.kurt(numeric_only=True,skipna=True)  # Calculate kurtosis across columns
            st.dataframe(kurt_df)
            plt.figure(figsize=(10, 4))
            plt.bar(kurt_df.index, kurt_df.values)
            plt.plot(skew_df.index.tolist(), skew_df.values.tolist(),color="red",marker="o",mec="black",mfc="blue")
            plt.xticks(rotation=45, ha='right')
            plt.title("Kurtosis (by Columns)")
            st.pyplot(plt.gcf())

        

class Krishna:
    def __init__(self,dataset):
        self.dataset=dataset
    def main(self):
        krishna.missing_plot(dataset)


        
class UnivariateWithoutHue:
     def __init__(self, dataset):
        self.dataset = dataset
        self.numeric_data_types = ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float16", "float32", "float64"]
        self.categorical_data_types = ["category", "object", "string", "datetime64[ns]", "bool"]
     def extract_columns(self):
        cc = self.dataset.select_dtypes(include=self.categorical_data_types, exclude=self.numeric_data_types).columns
        nc = self.dataset.select_dtypes(include=self.numeric_data_types, exclude=self.categorical_data_types).columns
        return cc, nc

     def plot_histplot(self, numerical_columns):
        plt.figure(figsize=(15, 5))
        for count, col in enumerate(numerical_columns, start=1):
            plt.subplot(1, len(numerical_columns), count)
            plt.title(f"Univariate Analysis with {col}")
            sns.histplot(data=self.dataset[col])
            plt.xlabel(col)
        plt.tight_layout()
        st.pyplot(plt)

     def plot_kdeplot(self, numerical_columns):
        plt.figure(figsize=(15, 5))
        for count, col in enumerate(numerical_columns, start=1):
            plt.subplot(1, len(numerical_columns), count)
            plt.title(f"Univariate Analysis with {col}")
            sns.kdeplot(data=self.dataset[col], fill=True)
            plt.xlabel(col)
        plt.tight_layout()
        st.pyplot(plt)

     def plot_boxplot(self, numerical_columns):
        plt.figure(figsize=(15, 5))
        for count, col in enumerate(numerical_columns, start=1):
            plt.subplot(1, len(numerical_columns), count)
            plt.title(f"Univariate Analysis with {col}")
            sns.boxplot(data=self.dataset[col])
            plt.xlabel(col)
        plt.tight_layout()
        st.pyplot(plt)

     def plot_violinplot(self, numerical_columns):
        plt.figure(figsize=(15, 5))
        for count, col in enumerate(numerical_columns, start=1):
            plt.subplot(1, len(numerical_columns), count)
            plt.title(f"Univariate Analysis with {col}")
            sns.violinplot(data=self.dataset[col])
            plt.xlabel(col)
        plt.tight_layout()
        st.pyplot(plt)

     def plot_stripplot(self, numerical_columns):
        plt.figure(figsize=(15, 5))
        for count, col in enumerate(numerical_columns, start=1):
            plt.subplot(1, len(numerical_columns), count)
            plt.title(f"Univariate Analysis with {col}")
            sns.stripplot(data=self.dataset[col])
            plt.xlabel(col)
        plt.tight_layout()
        st.pyplot(plt)

     def plot_swarmplot(self, numerical_columns):
        plt.figure(figsize=(15, 5))
        for count, col in enumerate(numerical_columns, start=1):
            plt.subplot(1, len(numerical_columns), count)
            plt.title(f"Univariate Analysis with {col}")
            sns.swarmplot(data=self.dataset[col])
            plt.xlabel(col)
        plt.tight_layout()
        st.pyplot(plt)

     def plot_ecdfplot(self, numerical_columns):
        plt.figure(figsize=(15, 5))
        for count, col in enumerate(numerical_columns, start=1):
            plt.subplot(1, len(numerical_columns), count)
            plt.title(f"Univariate Analysis with {col}")
            sns.ecdfplot(data=self.dataset[col])
            plt.xlabel(col)
        plt.tight_layout()
        st.pyplot(plt)

     def plot_rugplot(self, numerical_columns):
        plt.figure(figsize=(15, 5))
        for count, col in enumerate(numerical_columns, start=1):
            plt.subplot(1, len(numerical_columns), count)
            plt.title(f"Univariate Analysis with {col}")
            sns.rugplot(data=self.dataset[col])
            plt.xlabel(col)
        plt.tight_layout()
        st.pyplot(plt)

     def plot_lineplot(self, numerical_columns):
        plt.figure(figsize=(15, 5))
        for count, col in enumerate(numerical_columns, start=1):
            plt.subplot(1, len(numerical_columns), count)
            plt.title(f"Univariate Analysis with {col}")
            sns.lineplot(data=self.dataset[col])
            plt.xlabel(col)
        plt.tight_layout()
        st.pyplot(plt)

     def layout(self, nc):
        plot_dict = {
            'histplot': self.plot_histplot,
            'kdeplot': self.plot_kdeplot,
            'boxplot': self.plot_boxplot,
            'violinplot': self.plot_violinplot,
            'stripplot': self.plot_stripplot,
            'swarmplot': self.plot_swarmplot,
            'ecdfplot': self.plot_ecdfplot,
            'rugplot': self.plot_rugplot,
            'lineplot': self.plot_lineplot
        }
        
        if st.checkbox("Histplot With Out Hue"):
            plot_dict['histplot'](nc)
        if st.checkbox("KDE Plot With Out Hue"):
            plot_dict['kdeplot'](nc)
        if st.checkbox("Box Plot With Out Hue"):
            plot_dict['boxplot'](nc)
        if st.checkbox("Violin Plot With Out Hue"):
            plot_dict['violinplot'](nc)
        if st.checkbox("Strip Plot With Out Hue"):
            plot_dict['stripplot'](nc)
        if st.checkbox("Swarm Plot With Out Hue"):
            plot_dict['swarmplot'](nc)
        if st.checkbox("ECDF Plot With Out Hue"):
            plot_dict['ecdfplot'](nc)
        if st.checkbox("RUG Plot With Out Hue"):
            plot_dict['rugplot'](nc)
        if st.checkbox("Line Plot With Out Hue"):
            plot_dict['lineplot'](nc)

class UnivariateAnalysisWithHue:
    def __init__(self, data):
        self.dataset = data
        self.numeric_data_types = ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float16", "float32", "float64"]
        self.categorical_data_types = ["category", "object", "string", "bool"]
        
        # Initialize plot dictionary
        self.plot_dict = {
            'histplot': self.plot_histplot,
            'kdeplot': self.plot_kdeplot,
            'boxplot': self.plot_boxplot,
            'violinplot': self.plot_violinplot,
            'stripplot': self.plot_stripplot,
            'swarmplot': self.plot_swarmplot,
            'ecdfplot': self.plot_ecdfplot,
            'rugplot': self.plot_rugplot,
            'lineplot': self.plot_lineplot
        }
        self.value=st.slider("Select the hue features that contain at most given unique features in a particuler feature",min_value=1,max_value=100)
        
        # Extract categorical and numerical columns
        self.cc = [x for x in self.dataset.select_dtypes(include=self.categorical_data_types, exclude=self.numeric_data_types).columns if self.dataset[x].nunique()<=self.value]
        self.cc1 = [x for x in self.dataset.select_dtypes(include=self.categorical_data_types, exclude=self.numeric_data_types).columns if self.dataset[x].nunique()>self.value]
        self.nc = self.dataset.select_dtypes(include=self.numeric_data_types, exclude=self.categorical_data_types).columns

    def plot_histplot(self):
        plt.figure(figsize=(10, 25))
        cnt = 0
        for col in self.nc:
            for hue in self.cc:
                cnt += 1
                plt.subplot(len(self.nc), len(self.cc), cnt)
                sns.histplot(x=self.dataset[col],hue=self.dataset[hue])
                plt.xlabel(col)
        plt.tight_layout()
        st.pyplot(plt)

    def plot_kdeplot(self):
        plt.figure(figsize=(10, 25))
        cnt = 0
        for col in self.nc:
            for hue in self.cc:
                cnt += 1
                plt.subplot(len(self.nc), len(self.cc), cnt)
                sns.kdeplot(x=self.dataset[col],hue=self.dataset[hue], fill=True)
                plt.xlabel(col)
        plt.tight_layout()
        st.pyplot(plt)

    def plot_boxplot(self):
        plt.figure(figsize=(10, 25))
        cnt = 0
        for col in self.nc:
            for hue in self.cc:
                cnt += 1
                plt.subplot(len(self.nc), len(self.cc), cnt)
                sns.boxplot(x=self.dataset[col],hue=self.dataset[hue])
                plt.xlabel(col)
        plt.tight_layout()
        st.pyplot(plt)

    def plot_violinplot(self):
        plt.figure(figsize=(10, 25))
        cnt = 0
        for col in self.nc:
            for hue in self.cc:
                cnt += 1
                plt.subplot(len(self.nc), len(self.cc), cnt)
                sns.violinplot(x=self.dataset[col],hue=self.dataset[hue])
                plt.xlabel(col)
        plt.tight_layout()
        st.pyplot(plt)

    def plot_stripplot(self):
        plt.figure(figsize=(10, 25))
        cnt = 0
        for col in self.nc:
            for hue in self.cc:
                cnt += 1
                plt.subplot(len(self.nc), len(self.cc), cnt)
                sns.stripplot(data=self.dataset, x=col, hue=hue)
                plt.xlabel(col)
        plt.tight_layout()
        st.pyplot(plt)

    def plot_swarmplot(self):
        plt.figure(figsize=(10, 25))
        cnt = 0
        for col in self.nc:
            for hue in self.cc:
                cnt += 1
                plt.subplot(len(self.nc), len(self.cc), cnt)
                sns.swarmplot(x=self.dataset[col],hue=self.dataset[hue])
                plt.xlabel(col)
        plt.tight_layout()
        st.pyplot(plt)

    def plot_ecdfplot(self):
        plt.figure(figsize=(10, 25))
        cnt = 0
        for col in self.nc:
            for hue in self.cc:
                cnt += 1
                plt.subplot(len(self.nc), len(self.cc), cnt)
                sns.ecdfplot(x=self.dataset[col],hue=self.dataset[hue])
                plt.xlabel(col)
        plt.tight_layout()
        st.pyplot(plt)

    def plot_rugplot(self):
        plt.figure(figsize=(10, 25))
        cnt = 0
        for col in self.nc:
            for hue in self.cc:
                cnt += 1
                plt.subplot(len(self.nc), len(self.cc), cnt)
                sns.rugplot(x=self.dataset[col],hue=self.dataset[hue])
                plt.xlabel(col)
        plt.tight_layout()
        st.pyplot(plt)

    def plot_lineplot(self):
        plt.figure(figsize=(10, 25))
        cnt = 0
        for col in self.nc:
            for hue in self.cc:
                cnt += 1
                plt.subplot(len(self.nc), len(self.cc), cnt)
                sns.lineplot(x=self.dataset[col],hue=self.dataset[hue])
                plt.xlabel(col)
        plt.tight_layout()
        st.pyplot(plt)

    def layout(self):
        st.info(f"These values {self.cc1} have high cardinality, resulting in hundreds of plots, hence not included in plot generation")
        if st.checkbox("Histplot With Hue"):
            self.plot_dict['histplot']()
        if st.checkbox("KDE Plot With Hue"):
            self.plot_dict['kdeplot']()
        if st.checkbox("Box Plot With Hue"):
            self.plot_dict['boxplot']()
        if st.checkbox("Violin Plot With Hue"):
            self.plot_dict['violinplot']()
        if st.checkbox("Strip Plot With Hue"):
            self.plot_dict['stripplot']()
        if st.checkbox("Swarm Plot With Hue"):
            self.plot_dict['swarmplot']()
        if st.checkbox("ECDF Plot With Hue"):
            self.plot_dict['ecdfplot']()
        if st.checkbox("RUG Plot With Hue"):
            self.plot_dict['rugplot']()
        if st.checkbox("Line Plot With Hue"):
            self.plot_dict['lineplot']()

class AllPlots:
    def __init__(self, dataset):
        self.dataset = dataset
        self.numerical_columns = self.dataset.select_dtypes(include=["int8","int16","float16", "int32", "int64", "float32", "float64"])
        self.categorical_columns = self.dataset.select_dtypes(include=["category", "string", "object"])

    def relplot(self):
        for i in range(len(self.numerical_columns.columns)):
            for j in range(i + 1, len(self.numerical_columns.columns)):
                x_col = self.numerical_columns.columns[i]
                y_col = self.numerical_columns.columns[j]
                
                # Scatterplot
                st.write(f"### Scatterplot: {x_col} vs {y_col}")
                plt.figure(figsize=(10, 6))
                sns.scatterplot(data=self.dataset, x=x_col, y=y_col)
                plt.title(f"{x_col} vs {y_col}")
                st.pyplot(plt)
                
                # Lineplot
                st.write(f"### Lineplot: {x_col} vs {y_col}")
                plt.figure(figsize=(10, 6))
                sns.lineplot(data=self.dataset, x=x_col, y=y_col)
                plt.title(f"{x_col} vs {y_col}")
                st.pyplot(plt)
                
                # Relplot (which can do both scatter and line plots via kind parameter)
                st.write(f"### Relplot: {x_col} vs {y_col}")
                plt.figure(figsize=(10, 6))
                sns.relplot(data=self.dataset, x=x_col, y=y_col, kind="scatter")
                plt.title(f"{x_col} vs {y_col} - Scatter")
                st.pyplot(plt)
                
                plt.figure(figsize=(10, 6))
                sns.relplot(data=self.dataset, x=x_col, y=y_col, kind="line")
                plt.title(f"{x_col} vs {y_col} - Line")
                st.pyplot(plt)
    
    def distributions(self):
        for i in range(len(self.numerical_columns.columns)):
            for j in range(i + 1, len(self.numerical_columns.columns)):
                x_col = self.numerical_columns.columns[i]
                y_col = self.numerical_columns.columns[j]
                
                # Histplot (Bivariate)
                st.write(f"### Histplot: {x_col} vs {y_col}")
                plt.figure(figsize=(10, 6))
                sns.histplot(data=self.dataset, x=x_col, y=y_col)
                plt.title(f"Histogram: {x_col} vs {y_col}")
                st.pyplot(plt)
                
                # KDEplot (Bivariate)
                st.write(f"### KDEplot: {x_col} vs {y_col}")
                plt.figure(figsize=(10, 6))
                sns.kdeplot(data=self.dataset, x=x_col, y=y_col)
                plt.title(f"KDE: {x_col} vs {y_col}")
                st.pyplot(plt)
                
                # ECDFplot (Univariate)
                st.write(f"### ECDFplot: {x_col}")
                plt.figure(figsize=(10, 6))
                sns.ecdfplot(data=self.dataset, x=x_col)
                plt.title(f"ECDF: {x_col}")
                st.pyplot(plt)

                st.write(f"### ECDFplot: {y_col}")
                plt.figure(figsize=(10, 6))
                sns.ecdfplot(data=self.dataset, x=y_col)
                plt.title(f"ECDF: {y_col}")
                st.pyplot(plt)
                
                # Rugplot
                st.write(f"### Rugplot: {x_col} vs {y_col}")
                plt.figure(figsize=(10, 6))
                sns.rugplot(data=self.dataset, x=x_col, y=y_col)
                plt.title(f"Rugplot: {x_col} vs {y_col}")
                st.pyplot(plt)

    def regression_plots(self):
        for i in range(len(self.numerical_columns.columns)):
            for j in range(i + 1, len(self.numerical_columns.columns)):
                x_col = self.numerical_columns.columns[i]
                y_col = self.numerical_columns.columns[j]
            
                # lmplot
                try:
                    st.write(f"### lmplot: {x_col} vs {y_col}")
                    sns.lmplot(data=self.dataset, x=x_col, y=y_col, height=6, aspect=1.5)
                    plt.title(f"Linear Regression Model: {x_col} vs {y_col}")
                    st.pyplot(plt)
                except:
                    st.info(f"SOME ERROR GENERATED WHEN GENEARTING LMPLOT FOR {x_col} VS {y_col}")

                # regplot
                try:
                    st.write(f"### regplot: {x_col} vs {y_col}")
                    plt.figure(figsize=(10, 6))
                    sns.regplot(data=self.dataset, x=x_col, y=y_col)
                    plt.title(f"Regression Plot: {x_col} vs {y_col}")
                    st.pyplot(plt)
                except:
                    st.info(f"SOME ERROR GENERATED WHEN GENEARTING REGPLOT FOR {x_col} VS {y_col}")
                
                # residplot
                try:
                    st.write(f"### residplot: Residuals of {x_col} vs {y_col}")
                    plt.figure(figsize=(10, 6))
                    sns.residplot(data=self.dataset, x=x_col, y=y_col)
                    plt.title(f"Residual Plot: {x_col} vs {y_col}")
                    st.pyplot(plt)
                except:
                    st.info(f"SOME ERROR GENERATED WHEN GENEARTING RESIDPLOT FOR {x_col} VS {y_col}")
                    
    def matrix_plots(self):
        for i in range(len(self.numerical_columns.columns)):
            for j in range(i + 1, len(self.numerical_columns.columns)):
                x_col = self.numerical_columns.columns[i]
                y_col = self.numerical_columns.columns[j]

            # Compute the correlation matrix for the selected pair of columns
                corr_matrix = self.dataset[[x_col, y_col]].corr()

            # Heatmap
                try:
                    st.write(f"### Heatmap: {x_col} vs {y_col}")
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
                    plt.title(f"Heatmap of Correlation: {x_col} vs {y_col}")
                    st.pyplot(plt)
                except:
                    st.info()

            # Clustermap
                st.write(f"### Clustermap: {x_col} vs {y_col}")
                sns.clustermap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", figsize=(8, 6))
                plt.title(f"Clustermap of Correlation: {x_col} vs {y_col}")
                st.pyplot(plt)
    def multi_plot_grids(self):
    # Pairplot
        st.write("### Pairplot: Pairwise Relationships Between Numerical Variables")
        sns.pairplot(self.dataset[self.numerical_columns.columns])
        st.pyplot(plt)

    # PairGrid
        st.write("### PairGrid: Customized Pairwise Plots")
        pair_grid = sns.PairGrid(self.dataset[self.numerical_columns.columns])
        pair_grid.map_diag(sns.histplot)
        pair_grid.map_offdiag(sns.scatterplot)
        st.pyplot(pair_grid.fig)

    # Jointplot
        for i in range(len(self.numerical_columns.columns)):
            for j in range(i + 1, len(self.numerical_columns.columns)):
                x_col = self.numerical_columns.columns[i]
                y_col = self.numerical_columns.columns[j]

                st.write(f"### Jointplot: {x_col} vs {y_col}")
                sns.jointplot(data=self.dataset, x=x_col, y=y_col, kind="scatter", marginal_kws=dict(bins=15, fill=True))
                plt.title(f"Jointplot: {x_col} vs {y_col}", loc='left')
                st.pyplot(plt)

    # JointGrid
        for i in range(len(self.numerical_columns.columns)):
            for j in range(i + 1, len(self.numerical_columns.columns)):
                x_col = self.numerical_columns.columns[i]
                y_col = self.numerical_columns.columns[j]

                st.write(f"### JointGrid: Customized Jointplot for {x_col} vs {y_col}")
                joint_grid = sns.JointGrid(data=self.dataset, x=x_col, y=y_col)
                joint_grid.plot(sns.scatterplot, sns.histplot)
                st.pyplot(joint_grid.fig)
        st.divider()


class Cat_allPlots_num:
    def __init__(self, dataset):
        self.dataset = dataset
        self.numerical_columns = dataset.select_dtypes(include=["int8", "int32", "int64", "float32", "float64"])
        self.categorical_columns = dataset.select_dtypes(include=["category", "string", "object"]).copy()
        self.removed_columns = []

    def remove_high_cardinality_columns(self):
        """Removes categorical columns with more than 10 unique values."""
        for col in self.categorical_columns.columns:
            if self.categorical_columns[col].nunique() > 10:
                self.categorical_columns.drop(columns=col, inplace=True)
                self.removed_columns.append(col)

        if self.removed_columns:
            st.write(f"Dear user, we removed the following categorical columns from plotting due to their high cardinality: {self.removed_columns}")

    def relplot(self, hue):
        for i in range(len(self.numerical_columns.columns)):
            for j in range(i + 1, len(self.numerical_columns.columns)):
                x_col = self.numerical_columns.columns[i]
                y_col = self.numerical_columns.columns[j]

                # Scatterplot with Hue
                try:
                    st.write(f"### Scatterplot: {x_col} vs {y_col} (Categorical Hue: {hue})")
                    plt.figure(figsize=(10, 6))
                    sns.scatterplot(data=self.dataset, x=x_col, y=y_col, hue=hue)
                    plt.title(f"Scatterplot: {x_col} vs {y_col} with {hue} Hue")
                    st.pyplot(plt)
                except:
                    st.info(f"An error occurred when generating scatterplot for {x_col} vs {y_col} with {hue} Hue.")

                # Lineplot with Hue
                try:
                    st.write(f"### Lineplot: {x_col} vs {y_col} (Categorical Hue: {hue})")
                    plt.figure(figsize=(10, 6))
                    sns.lineplot(data=self.dataset, x=x_col, y=y_col, hue=hue)
                    plt.title(f"Lineplot: {x_col} vs {y_col} with {hue} Hue")
                    st.pyplot(plt)
                except:
                    st.info(f"An error occurred when generating lineplot for {x_col} vs {y_col} with {hue} Hue.")

    def main(self):
        self.remove_high_cardinality_columns()

        # Loop through each pair of categorical columns and call self.relplot() for each pair
        for col in self.categorical_columns.columns:
            for hue in self.categorical_columns.columns:
                if col != hue:
                    self.relplot(hue)

class Cat_Cat:
    def __init__(self, dataset):
        self.dataset = dataset
        self.categorical_columns = dataset.select_dtypes(include=["category", "string", "object"]).copy()
        self.removed_columns = []

    def remove_high_cardinality_columns(self):
        """Remove categorical columns with more than 10 unique values."""
        for col in self.categorical_columns.columns:
            if self.categorical_columns[col].nunique() > 10:
                self.categorical_columns.drop(columns=col, inplace=True)
                self.removed_columns.append(col)
        
        if self.removed_columns:
            st.write(f"Dear user, we removed the following categorical columns from plotting due to their high cardinality: {self.removed_columns}")

    def count_plot(self, col_x, col_y):
        st.write(f"### Count Plot: {col_x} vs {col_y}")
        sns.countplot(data=self.dataset, x=col_x, hue=col_y)
        st.pyplot(plt)
        plt.clf()

    def heatmap_plot(self, col_x, col_y):
        st.write(f"### Heatmap: {col_x} vs {col_y}")
        cross_tab = pd.crosstab(self.dataset[col_x], self.dataset[col_y])
        sns.heatmap(cross_tab, annot=True, fmt="d")
        st.pyplot(plt)
        plt.clf()

    def point_plot(self, col_x, col_y):
        st.write(f"### Point Plot: {col_x} vs {col_y}")
        sns.pointplot(data=self.dataset, x=col_x, y=col_y)
        st.pyplot(plt)
        plt.clf()

    def boxen_plot(self, col_x, col_y):
        st.write(f"### Boxen Plot: {col_x} vs {col_y}")
        sns.boxenplot(data=self.dataset, x=col_x, y=col_y)
        st.pyplot(plt)
        plt.clf()

    def strip_plot(self, col_x, col_y):
        st.write(f"### Strip Plot: {col_x} vs {col_y}")
        sns.stripplot(data=self.dataset, x=col_x, y=col_y)
        st.pyplot(plt)
        plt.clf()

    def violin_plot(self, col_x, col_y):
        st.write(f"### Violin Plot: {col_x} vs {col_y}")
        sns.violinplot(data=self.dataset, x=col_x, y=col_y)
        st.pyplot(plt)
        plt.clf()

    def swarm_plot(self, col_x, col_y):
        st.write(f"### Swarm Plot: {col_x} vs {col_y}")
        sns.swarmplot(data=self.dataset, x=col_x, y=col_y)
        st.pyplot(plt)
        plt.clf()

    def pairplot(self, col_x, col_y):
        st.write(f"### Pairplot: {col_x} vs {col_y}")
        sns.pairplot(self.dataset, hue=col_x)
        st.pyplot(plt)
        plt.clf()

    def main(self):
        self.remove_high_cardinality_columns()

        if len(self.categorical_columns.columns) == 0:
            st.write("No categorical columns available for plotting after filtering high cardinality columns.")
            return

        for i in range(len(self.categorical_columns.columns)):
            for j in range(i + 1, len(self.categorical_columns.columns)):
                col_x = self.categorical_columns.columns[i]
                col_y = self.categorical_columns.columns[j]
                
                self.count_plot(col_x, col_y)
                self.heatmap_plot(col_x, col_y)
                self.point_plot(col_x, col_y)
                self.boxen_plot(col_x, col_y)
                self.strip_plot(col_x, col_y)
                self.violin_plot(col_x, col_y)
                self.swarm_plot(col_x, col_y)
                self.pairplot(col_x, col_y)



                   
# Sidebar for file upload and menu
csv_file = st.sidebar.file_uploader("Upload Any CSV File", type=["csv"])
with st.sidebar:
    option_menus = option_menu("Analyser Menu", ["Pandas Basic Informative Dashboard",  "Univariate Analysis",
                                                "Hundred's of plots"])

# Check if a CSV file is uploaded
if csv_file:
    # Reading the uploaded file as bytes (file-like object)
    csv_bytes = csv_file.read(100000)  # Read the first 100KB to detect encoding
    result = chardet.detect(csv_bytes)
    encoding = result['encoding']

    # Move back to the start of the file after reading
    csv_file.seek(0)

    # Read CSV using the detected encoding
    dataframe = pd.read_csv(csv_file, encoding=encoding)

    # Assuming `krishna` is an instance of a class that contains the method `data_cleaning`.
    value = krishna.data_cleaning(dataframe)

    # Option for Pandas Basic Informative Dashboard
    if option_menus == "Pandas Basic Informative Dashboard":
        pandas = Statistics(value)
        pandas.basic_details()
        pandas.secondary_information()
        pandas.statistics_1()
        pandas.statistics_2()

    # Option for Univariate Analysis
    elif option_menus == "Univariate Analysis":
        with st.expander("Univariate Analysis - Basic"):
            univariateAnalysis = UnivariateWithoutHue(value)
            cc, nc = univariateAnalysis.extract_columns()
            univariateAnalysis.layout(nc)

        with st.expander("Univariate Analysis - Intermediate"):
            uWh = UnivariateAnalysisWithHue(value)
            uWh.layout()

    elif option_menus == "Hundred's of plots":
        all_plots_instance = AllPlots(value)
        col1,col2=st.columns([1,2])
        with col1:
            if st.checkbox("Apply ALL Rel Plots"):
                with col2:
                    all_plots_instance.relplot()
            if st.checkbox("Apply ALL Distribution Plots"):
                with col2:
                    all_plots_instance.distributions()
            if st.checkbox("Apply ALL Regression Plots"):
                with col2:
                    all_plots_instance.regression_plots()
            if st.checkbox("Apply ALL Matrix Plots"):
                with col2:
                    all_plots_instance.matrix_plots()
            if st.checkbox("Apply ALL Multi Plot grids"):
                with col2:
                    all_plots_instance.multi_plot_grids()
            if st.checkbox("Apply ALL Categorical Plots"):
                with col2:
                    cat_plots_instance = Cat_allPlots_num(value)
                    cat_plots_instance.main()
            if st.checkbox("Apply ALL Categorical Plots VS CAtegorical Plots"):
                with col2:
                    Cat_Cat(value).main()
