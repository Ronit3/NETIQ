import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sb
from tkinter import Tk, Canvas, Button, PhotoImage, filedialog, Frame, messagebox
from pathlib import Path
from pandastable import Table
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / "assets" / "frame0"

fileloc = ''
outp = 'Welcome to NetIQ - Credits: S Meghanath Reddy, K Purnanandh, G Peyush. VIT CHENNAI'
df = pd.DataFrame()

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def fileOpen():
    global fileloc, outp, df
    loc = filedialog.askopenfilename(filetypes=[("csv files", "*.csv")])
    if loc:
        fileloc = loc
        outp = f'File selected from location {fileloc}'
        canvas.itemconfig(output_text, text=outp)
        try:
            df = pd.read_csv(fileloc)
            table = Table(table_frame, dataframe=df)
            table.show()
        except Exception as e:
            messagebox.showerror('Error', f'Failed to load CSV file: {e}')
            fileloc = ''
            outp = 'File not selected'
            canvas.itemconfig(output_text, text=outp)

def prepBal():
    global fileloc, outp, df
    if not fileloc:
        messagebox.showerror('Error', 'Please select a file first')
        return
    try:
        df = pd.read_csv(fileloc)
        outp = 'Selected file for preprocessing'
        canvas.itemconfig(output_text, text=outp)
        
        # Fill missing values
        df.fillna(0, inplace=True)
        outp = 'Missing values are filled'
        canvas.itemconfig(output_text, text=outp)

        # Encode categorical features
        outp = 'Encoding categorical features'
        canvas.itemconfig(output_text, text=outp)
        def convert_object_columns(df):
            label_encoders = {}
            for col in df.columns:
                if df[col].dtype == 'object':
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    label_encoders[col] = le
            return df
        dfr = convert_object_columns(df)

        # Assume 'label' is the target column
        X = df.drop(columns=['label'])
        y = df['label']
        
        outp = 'Balancing the dataset'
        canvas.itemconfig(output_text, text=outp)
        
        # Apply RandomUnderSampler to undersample the majority class
        rus = RandomUnderSampler(sampling_strategy='auto')
        X_resampled, y_resampled = rus.fit_resample(X, y)

        # Combine resampled features and target into a DataFrame
        dfr = pd.DataFrame(X_resampled)
        dfr['label'] = y_resampled
        df = dfr
        outp = 'Preprocessing and Balancing completed'
        canvas.itemconfig(output_text, text=outp)

        table = Table(table_frame, dataframe=df)
        table.show()
    except Exception as e:
        messagebox.showerror('Error', f'Error in preprocessing: {e}')

def clear_frame(frame):
    for widget in frame.winfo_children():
        widget.destroy()

def rForest():
    global df, fileloc, outp
    if fileloc == '':
        messagebox.showerror('Error', 'Please select a file first and preprocess it')
        return
    
    try:
        outp = 'Random Forest classification initiated'
        canvas.itemconfig(output_text, text=outp)
        
        X = df.drop(columns=['label'])
        y = df['label']
        
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        rf_model = RandomForestClassifier(n_estimators=20, random_state=42)
        rf_model.fit(x_train, y_train)
        rf_predictions = rf_model.predict(x_test)
        rf_accuracy = accuracy_score(y_test, rf_predictions)
        outp = f'Random Forest Accuracy Score: {rf_accuracy}'
        canvas.itemconfig(output_text, text=outp)
        
        # Create a confusion matrix
        rf_cm = confusion_matrix(y_test, rf_predictions)
    
        # Clear the frame before displaying the plot
        clear_frame(table_frame)
        
        # Plot the confusion matrix
        fig, ax = plt.subplots()
        sb.heatmap(rf_cm, annot=True, cmap='Blues', fmt='g', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix for Random Forest Classifier')
        
        # Create a Matplotlib figure canvas and embed it in the tkinter window
        canvas_plt = FigureCanvasTkAgg(fig, master=table_frame)
        canvas_plt.draw()
        canvas_plt.get_tk_widget().pack()
    
    except Exception as e:
        messagebox.showerror('Error', f'Error in Random Forest classification: {e}')

def knn():
    global df, fileloc, outp
    if fileloc == '':
        messagebox.showerror('Error', 'Please select a file first and preprocess it')
        return
    
    try:
        outp = 'KNN classification initiated'
        canvas.itemconfig(output_text, text=outp)
        
        X = df.drop(columns=['label'])
        y = df['label']
        
        X = (X - np.min(X))/(np.max(X)-np.min(X))
        
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(x_train, y_train)
        test_prediction = knn.predict(x_test)
        outp = f"kNN (k = 3) accuracy score: {knn.score(x_test, y_test)}"
        canvas.itemconfig(output_text, text=outp)
        k_cm = confusion_matrix(y_test, test_prediction)
        
        # Clear the frame before displaying the plot
        clear_frame(table_frame)
        
        # Plot the confusion matrix
        fig, ax = plt.subplots()
        sb.heatmap(k_cm, annot=True, cmap='Reds', fmt='g', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix for kNN Classifier')
        
        # Create a Matplotlib figure canvas and embed it in the tkinter window
        canvas_plt = FigureCanvasTkAgg(fig, master=table_frame)
        canvas_plt.draw()
        canvas_plt.get_tk_widget().pack()    
        
    except Exception as e:
        messagebox.showerror('Error', f'Error in kNN classification: {e}')

def svm():
    global df, fileloc, outp
    if fileloc == '':
        messagebox.showerror('Error', 'Please select a file first and preprocess it')
        return
    
    try:
        outp = 'SVM classification initiated'
        canvas.itemconfig(output_text, text=outp)
        
        X = df.drop(columns=['label'])
        y = df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=74)
        model2 = SVC(kernel='rbf')
        model2.fit(X_train, y_train)
        test_prediction = model2.predict(X_test)
        k_cm = confusion_matrix(y_test, test_prediction)
        svm_accuracy = model2.score(X_test, y_test)
        outp = f'SVM Accuracy Score: {svm_accuracy}'
        canvas.itemconfig(output_text, text=outp)
        
        # Clear the frame before displaying the plot
        clear_frame(table_frame)        
        
        # Plot the confusion matrix
        fig, ax = plt.subplots()
        sb.heatmap(k_cm, annot=True, fmt='g', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix for SVM Classifier')
        # Create a Matplotlib figure canvas and embed it in the tkinter window
        canvas_plt = FigureCanvasTkAgg(fig, master=table_frame)
        canvas_plt.draw()
        canvas_plt.get_tk_widget().pack()
    
    except Exception as e:
        messagebox.showerror('Error', f'Error in SVM classification: {e}')

def naiveB():
    global df, fileloc, outp
    if fileloc == '':
        messagebox.showerror('Error', 'Please select a file first and preprocess it')
        return
    try:
        outp = 'Naive Bayes classification initiated'
        canvas.itemconfig(output_text, text=outp)
        
        X = df.drop(columns=['label'])
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Multinomial Naive Bayes
        mnb = MultinomialNB()
        mnb.fit(X_train, y_train)

        # Predict on the test set
        y_pred_mnb = mnb.predict(X_test)
        outp = f'Naive Bayes Accuracy Score: {accuracy_score(y_test, y_pred_mnb)}'
        canvas.itemconfig(output_text, text=outp)

        # Confusion Matrix for Multinomial NB
        nb_cm = confusion_matrix(y_test, y_pred_mnb)  
        # Clear the frame before displaying the plot
        clear_frame(table_frame)
        
        # Plot the confusion matrix
        fig, ax = plt.subplots()
        sb.heatmap(nb_cm, annot=True, cmap='Greens', fmt='g', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix for Naive Bayes Classifier')
        
        # Create a Matplotlib figure canvas and embed it in the tkinter window
        canvas_plt = FigureCanvasTkAgg(fig, master=table_frame)
        canvas_plt.draw()
        canvas_plt.get_tk_widget().pack()    
        
    except Exception as e:
        messagebox.showerror('Error', f'Error in Naive Bayes classification: {e}')

window = Tk()

window.geometry("900x700")
window.title("NetIQ - Network Classifier")
window.configure(bg="#8ECAE6")

canvas = Canvas(
    window,
    bg="#8ECAE6",
    height=700,
    width=900,
    bd=0,
    highlightthickness=0,
    relief="ridge"
)

canvas.place(x=0, y=0)
canvas.create_rectangle(
    0.0,
    0.0,
    900.0,
    58.0,
    fill="#023047",
    outline="")

image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    41.0,
    29.0,
    image=image_image_1
)

image_image_2 = PhotoImage(
    file=relative_to_assets("image_2.png"))
image_2 = canvas.create_image(
    860.0,
    29.0,
    image=image_image_2
)

canvas.create_text(
    175.0,
    13.0,
    anchor="nw",
    text="NetIQ - Network Traffic Classifier",
    fill="#FFFFFF",
    font=("Quicksand Bold", 24 * -1)
)

button_1 = Button(
    text='Upload CSV',
    borderwidth=0,
    highlightthickness=0,
    command=fileOpen,
    relief="flat"
)
button_1.place(
    x=19.0,
    y=74.0,
    width=409.0,
    height=42.0
)

button_2 = Button(
    text='Preprocess and Balance',
    borderwidth=0,
    highlightthickness=0,
    command=prepBal,
    relief="flat"
)
button_2.place(
    x=474.0,
    y=74.0,
    width=409.0,
    height=42.0
)

button_3 = Button(
    text='kNN',
    borderwidth=0,
    highlightthickness=0,
    command=knn,
    relief="flat"
)
button_3.place(
    x=19.0,
    y=144.0,
    width=184.0,
    height=42.0
)

button_4 = Button(
    text='Naive Bayes',
    borderwidth=0,
    highlightthickness=0,
    command=naiveB,
    relief="flat"
)
button_4.place(
    x=243.0,
    y=144.0,
    width=185.0,
    height=42.0
)

button_5 = Button(
    text='SVM',
    borderwidth=0,
    highlightthickness=0,
    command=svm,
    relief="flat"
)
button_5.place(
    x=474.0,
    y=144.0,
    width=185.0,
    height=42.0
)

button_6 = Button(
    text='Random Forest',
    borderwidth=0,
    highlightthickness=0,
    command=rForest,
    relief="flat"
)
button_6.place(
    x=699.0,
    y=144.0,
    width=184.0,
    height=42.0
)

canvas.create_text(
    19.0,
    74.0,
    anchor="nw",
    text="Upload CSV",
    fill="#000000",
    font=("Quicksand Bold", 14 * -1)
)

canvas.create_text(
    474.0,
    74.0,
    anchor="nw",
    text="Preprocess and balance",
    fill="#000000",
    font=("Quicksand Bold", 14 * -1)
)

canvas.create_text(
    19.0,
    144.0,
    anchor="nw",
    text="K-NN",
    fill="#000000",
    font=("Quicksand Bold", 16 * -1)
)

canvas.create_text(
    243.0,
    144.0,
    anchor="nw",
    text="Naive Bayes",
    fill="#000000",
    font=("Quicksand Bold", 16 * -1)
)

canvas.create_text(
    474.0,
    144.0,
    anchor="nw",
    text="SVM",
    fill="#000000",
    font=("Quicksand Bold", 16 * -1)
)

canvas.create_text(
    699.0,
    144.0,
    anchor="nw",
    text="Random Forest",
    fill="#000000",
    font=("Quicksand Bold", 16 * -1)
)

canvas.create_rectangle(
    18.0,
    206.0,
    882.0,
    258.0,
    fill="#FFB703",
    outline="")

canvas.create_rectangle(
    19.0,
    278.0,
    882.0,
    680.0,
    fill="#D9D9D9",
    outline="")

output_text = canvas.create_text(
    450.0,
    225.0,
    anchor="center",
    text=outp,
    fill="#000000",
    font=("Quicksand Bold", 16 * -1)
)

table_frame = Frame(canvas)
table_frame.place(x=0, y=278, width=900, height=680)  # Adjust the coordinates and size as needed

window.resizable(False, False)
window.mainloop()
