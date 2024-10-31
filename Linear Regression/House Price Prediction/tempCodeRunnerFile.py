plt.scatter(df.iloc[:, 3], df.iloc[:, 6], color='blue', label='Data Points')

# # Fit a line using NumPy's polyfit (degree=1 for linear fit)
# coefficients = np.polyfit(df.iloc[:, 3], df.iloc[:, 6], 1)  # Linear regression
# linear_fit = np.poly1d(coefficients)

# # Plot the regression line
# x = np.linspace(df.iloc[:, 3].min(), df.iloc[:, 3].max(), 100)
# plt.plot(x, linear_fit(x), color='red', label='Linear Fit')

# # Add labels and legend
# plt.xlabel('Feature')
# plt.ylabel('Target')
# plt.legend()
# plt.show()
# correlation = df.iloc[:, 3].corr(df.iloc[:, 6])
# print(f"Correlation coefficient: {correlation}")
