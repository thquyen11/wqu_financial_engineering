using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Module2_PRA
{
    class PayrollRunner
    {
        static void Main(string[] args)
        {
            // use Employee without tax
            Employee john = new Employee(1, "John Doe", 20000, false);
            john.printInformation();

            // use Employee with tax
            Employee jane = new Employee(2, "Jane Doe", 36000);
            jane.printInformation();

            // use WeeklyEmployee without tax
            WeeklyEmployee jack = new WeeklyEmployee(3, "Jack Deer", 18500, false);
            jack.printInformation();

            // use WeeklyEmployee with tax
            WeeklyEmployee jen = new WeeklyEmployee(4, "Jen Deer", 18000);
            jen.printInformation();

            Console.Read();
        }
    }

    class Employee
    {
        protected int employeeId { get;set;}
        protected string fullName { get;set;}
        protected float salary { get;set;}
        protected bool taxDeducted { get;set;}
        public Employee(int employeeId, string fullName, float salary, bool taxDeducted)
        {
            this.employeeId = employeeId;
            this.fullName = fullName;
            this.salary = salary;
            this.taxDeducted = taxDeducted;
        }

        public Employee(int employeeId, string fullName, float salary)
        {
            this.employeeId = employeeId;
            this.fullName = fullName;
            this.salary = salary;
            this.taxDeducted = true;
        }

        private double getNetSalary(double taxDeductedPct)
        {
            return (this.taxDeducted == true ? (1-taxDeductedPct)*this.salary : this.salary);
        }

        public void printInformation()
        {
            Console.WriteLine(String.Format("{0}, {1} earns {2} per month.",this.employeeId,this.fullName, getNetSalary(0.2)));
        }
    }

    class WeeklyEmployee : Employee
    {
        public WeeklyEmployee(int employeeId, string fullName, float salary, bool taxDeducted): base(employeeId, fullName, salary, taxDeducted){}
        public WeeklyEmployee(int employeeId, string fullName, float salary): base(employeeId, fullName, salary){}

        private double getNetSalary(double taxDeductedPct)
        {
            return this.taxDeducted == true ? (1-taxDeductedPct)*this.salary/4 : this.salary/4;
        }

        public void printInformation()
        {
            Console.WriteLine(String.Format("{0}, {1} earns {2} per week.",this.employeeId,this.fullName, getNetSalary(0.2)));
        }
    }
}
