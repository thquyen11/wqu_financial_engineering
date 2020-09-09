using System;
using System.Linq;
using System.Collections.Generic;
using Excel = Microsoft.Office.Interop.Excel;
using Microsoft.Office.Interop.Excel;
using System.Globalization;

namespace WorldQuant_Module3_CSA_SkeletonCode
{
    class Program
    {
        static Excel.Workbook workbook;
        static Excel.Application app;
        static Excel._Worksheet currentSheet;

        static void Main(string[] args)
        {
            app = new Excel.Application();
            app.Visible = true;
            int rows = 2; //Skip the header in 1st row
            try
            {
                workbook = app.Workbooks.Open("property_pricing.xlsx", ReadOnly: false);
            }
            catch
            {
                SetUp();
            }

            // insert header in excel file
            currentSheet = workbook.Worksheets[1];
            currentSheet.Cells[1, "A"] = "Size";
            currentSheet.Cells[1, "B"] = "Suburb";
            currentSheet.Cells[1, "C"] = "City";
            currentSheet.Cells[1, "D"] = "Value";

            var input = "";
            while (input != "x")
            {
                PrintMenu();
                input = Console.ReadLine();
                try
                {
                    var option = int.Parse(input);
                    switch (option)
                    {
                        case 1:
                            try
                            {
                                Console.Write("Enter the size: ");
                                var size = float.Parse(Console.ReadLine());
                                Console.Write("Enter the suburb: ");
                                var suburb = Console.ReadLine();
                                Console.Write("Enter the city: ");
                                var city = Console.ReadLine();
                                Console.Write("Enter the market value: ");
                                var value = float.Parse(Console.ReadLine());

                                // count the number of row
                                AddPropertyToWorksheet(size, suburb, city, value, rows);
                                rows++;
                            }
                            catch
                            {
                                Console.WriteLine("Error: couldn't parse input");
                            }
                            break;
                        case 2:
                            Console.WriteLine("Mean price: " + CalculateMean("D"));
                            break;
                        case 3:
                            Console.WriteLine("Price variance: " + CalculateVariance("D"));
                            break;
                        case 4:
                            Console.WriteLine("Minimum price: " + CalculateMinimum("D"));
                            break;
                        case 5:
                            Console.WriteLine("Maximum price: " + CalculateMaximum("D"));
                            break;
                        default:
                            break;
                    }
                } catch { }
            }

            // save before exiting
            workbook.SaveAs("property_pricing.xlsx");
            workbook.Close();
            app.Quit();
        }

        static void PrintMenu()
        {
            Console.WriteLine();
            Console.WriteLine("Select an option (1, 2, 3, 4, 5) " +
                              "or enter 'x' to quit...");
            Console.WriteLine("1: Add Property");
            Console.WriteLine("2: Calculate Mean");
            Console.WriteLine("3: Calculate Variance");
            Console.WriteLine("4: Calculate Minimum");
            Console.WriteLine("5: Calculate Maximum");
            Console.WriteLine();
        }

        static void SetUp()
        {
            workbook  = app.Workbooks.Add(XlWBATemplate.xlWBATWorksheet);            
        }

        static void AddPropertyToWorksheet(float size, string suburb, string city, float value, int rows)
        {           
            currentSheet.Cells[rows, "A"] = size;
            currentSheet.Cells[rows, "B"] = suburb;
            currentSheet.Cells[rows, "C"] = city;
            currentSheet.Cells[rows, "D"] = value;
        }

        static float CalculateMean(string column)
        {
            List<float> listMV = ColumnToList(column);
            if(listMV == null || !listMV.Any())
            {
                Console.WriteLine("Empty data, please insert at least 1 row of data");
                return 0;
            }
     
            return listMV.Sum() / listMV.Count();
        }

        static double CalculateVariance(string column)
        {
            List<float> listMV = ColumnToList(column);
            if (listMV == null || !listMV.Any())
            {
                Console.WriteLine("Empty data, please insert at least 1 row of data");
                return 0;
            }

            double avg = CalculateMean("D");
            double sumOfSquares = 0.0;

            foreach (int num in listMV)
            {
                sumOfSquares += Math.Pow((num - avg), 2);
            }

            return sumOfSquares / (listMV.Count()-1);
        }

        static float CalculateMinimum(string column)
        {
            List<float> listMV = ColumnToList(column);
            if (listMV == null || !listMV.Any())
            {
                Console.WriteLine("Empty data, please insert at least 1 row of data");
                return 0;
            }

            listMV.Sort();
            return listMV[0];
        }

        static float CalculateMaximum(string column)
        {
            List<float> listMV = ColumnToList(column);
            if (listMV == null || !listMV.Any())
            {
                Console.WriteLine("Empty data, please insert at least 1 row of data");
                return 0;
            }

            listMV.Sort();
            return listMV[listMV.Count()-1];
        }

        static List<float> ColumnToList(string column)
        {
            int i = 2;
            string cellValue;
            Excel.Range rangeObj;
            List<float> dataItems = new List<float>();

            do
            {
                rangeObj = currentSheet.Cells[i, column] as Excel.Range;
                if (rangeObj.Value2 is null) break;
                cellValue = rangeObj.Value2.ToString();
                dataItems.Add(float.Parse(cellValue));
                i++;
            } while (true);

            return dataItems;
        }
    }
}
