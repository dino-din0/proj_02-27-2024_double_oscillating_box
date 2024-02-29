#-------------------------------------------------------
"""

    This program solves an arbitrary set of first-order
    ordinary differential equations

    All that the user would need to do is to implement the right-hand sides
    of the differential equations in the 'ode_sys' function and the
    right-hand side jacobians in the 'jacob' function, both in the
    My_Differential_System() class

    The parameters of the differential equations are 
    stored in a 'self.par[][]' array
    
    The program also calculates eigenvalues

    The remaining pieces of needed information are requested of the user at run time
"""
#-------------------------------------------------------
#-------------------------------------------------------
import numpy
import math # it may be needed for some differential equations
from   scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
import matplotlib.animation as animation

#-------------------------------------------------------
#-------------------------------------------------------
def main():
      
    # Class objects
    eigenvalues     = Eigenvalue_Sets()       
    diff_sys        = My_Differential_System( eigenvalues.choice )   
    
    if ( eigenvalues.choice != 'yes' ):
    
    
        solve_system    = Solve_Differential_System()
                 
        # object utilization
        solve_system.solve_sys( diff_sys )
          
        solve_system.print_to_file()
        #solve_system.print_to_screen()
        
        my_plot = Plot()
        
        my_plot.chart_it( solve_system.data )
        
        coordinates     = Define_Points ( diff_sys.number_of_moving_bodies, solve_system.comp_sol ) 
        coordinates.point_coordinates()
        
        action = Perform_Animation( diff_sys.delta_t, coordinates ) 
        action.set_up_animation()
        
    else:
        
        eigenvalues.calculate_eigenvalues( diff_sys )
   
        eigenvalues.print_to_screen()
        eigenvalues.print_to_file()
        


    return
#-------------------------------------------------------

#-------------------------------------------------------
class Define_Differential_System( object ):

    #---------------------------------------
    def __init__( self ):

        #self.number_of_equations_and_parameters()
        #self.init_cond_form()
        
    
        self.time_set_form()
        
        return
    #---------------------------------------
    #---------------------------------------
    def number_of_equations_and_parameters( self ):

        print()
        self.numb_of_eq     = int( input('Enter number of equations: ') )
        self.par = []
        
                
        info = str( input( 'Are parameters needed in this system? Enter yes or no: ' ) )

        if ( info == 'yes'):
            
            par_matrx = []
            for i in range( self.numb_of_eq ):
                
                print()
                print('Equation ', i + 1 )
                numb_of_par = abs( int( input('Enter number of parameters for this equation: ') ) )
                
                a = []
                for j in range( numb_of_par ):
                    
                    print('Parameter No. ', j + 1 , ' - enter this parameter: ', end = '')
                    a.append( float( input() ) )
                                
                par_matrx.append( a )
        
            a = len( par_matrx[ 0 ] )
            for i in range( 1, self.numb_of_eq ):
                if( len( par_matrx[ i ] ) > a ):
                    a = len( par_matrx[ i ] )
                        
            self.par = numpy.zeros( ( self.numb_of_eq, a ), dtype = float )  
            
            
            for i in range( self.numb_of_eq ):
                a = len( par_matrx[ i ] )
                for j in range( a ) :
                    self.par[ i ][ j ] =  par_matrx[ i ][ j ]
        
        return

    #---------------------------------------
    def init_cond_form( self ):
    
        self.q_in        =  numpy.zeros(  self.numb_of_eq, dtype = float )
       
        print()
        for i in range( self.numb_of_eq ):
            print('Initial condition No. ', i + 1, ' - enter this condition: ', end='')
            self.q_in[ i ] = float( input() )

        return

    #---------------------------------------
    #---------------------------------------
    def init_cond( self ):
     
        return( self.q_in )

    #---------------------------------------
    def time_set_form( self ):
    
        print()
        self.t_begin     =  float( input( 'Enter initial time: ' ) )
        self.t_end       =  float( input( 'Enter final time: ' ) )
        self.delta_t     =  float( input( 'Enter time step Dt: ') )
        
        
        self.t_stations = int( ( self.t_end - self.t_begin ) / self.delta_t )
        
        t_temp = self.delta_t * float( self.t_stations ) + self.t_begin      
        if ( t_temp < self.t_end ):
            self.t_end = t_temp + self.delta_t
        else :
            self.t_end = t_temp
            
        print()
        print( "Number of time stations: ", self.t_stations )
        print( "Final time: ", self.t_end )
        input()

        self.time = numpy.linspace( self.t_begin, self.t_end, self.t_stations )

        self.time_range = numpy.zeros( 2, dtype = float )
        self.time_range[ 0 ] = self.t_begin
        self.time_range[ 1 ] = self.t_end

        return

    #---------------------------------------
    def time_set( self ):
    
        return( self.time_range, self.time )
    
#-------------------------------------------------------
#-------------------------------------------------------
class My_Differential_System( Define_Differential_System ):
    
    #---------------------------------------
    def __init__( self, choice ):
        
        self.eig_par = 0.0
        self.choice = choice
        self.number_of_equations_and_parameters_2()
        self.init_cond_form_2()

        if ( self.choice != "yes" ):
            Define_Differential_System.__init__( self )
        
        
    #--------------------------------------- 
    def number_of_equations_and_parameters_2( self ):
        
        self.numb_of_eq      = 4
        self.numb_of_par     = 7
        self.number_of_moving_bodies = 2
        #-------------------------------------------
        # System parameters
                    
        k = 16.0
        
        xi = 0.0 # 0.0 for worst case scenario
           
        k_2_k_1 = 0.05
        
        m_2_m_1 = 0.1
        
        c_2_c_1 = 0.05
        
        
        ell_0 = 1.0 # Unstretched length of the second spring
               
        fs_mag =  1.0 # Frequency
        
        omega = 2.706394092 # Set omega to 1.0 if we don't know what the natrl. frequency is
        # From the eigenvalue analysis, natrual frequencies are: 4.180362546 and 2.706394092



        
        #-------------------------------------------
            
        #-------------------------------------------   

        if ( self.choice == "yes" ):
            xi = self.eig_par

        self.par = numpy.zeros( ( self.numb_of_eq, self.numb_of_par ), dtype = float ) 

        self.par[ 1 ][ 0 ] = k * ( 1.0 + k_2_k_1 )
        self.par[ 1 ][ 1 ] = 2.0 * xi * math.sqrt( k ) * ( 1.0 + c_2_c_1 )
        self.par[ 1 ][ 2 ] = k * k_2_k_1
        self.par[ 1 ][ 3 ] = 2.0 * xi * math.sqrt( k ) * c_2_c_1 
        
        self.par[ 1 ][ 4 ] = ell_0
        
        
        self.par[ 1 ][ 5 ] = fs_mag 
        self.par[ 1 ][ 6 ] = omega
        
        self.par[ 3 ][ 0 ] = k * k_2_k_1 / m_2_m_1
        self.par[ 3 ][ 1 ] = 2.0 * xi * math.sqrt( k ) * c_2_c_1 / m_2_m_1
        self.par[ 3 ][ 2 ] = k * k_2_k_1 / m_2_m_1  
        self.par[ 3 ][ 3 ] = 2.0 * xi * math.sqrt( k ) * c_2_c_1 / m_2_m_1
        
        self.par[ 3 ][ 4 ] = ell_0        
        
       
        self.par[ 3 ][ 2 ] = self.par[ 1 ][ 6 ]**2.0 # This line eliminates resonance. If we want to find resonance, we comment this out
        
        
        return
    #---------------------------------------
    def init_cond_form_2( self ):
    
        self.q_in      =  numpy.zeros(  self.numb_of_eq, dtype = float )
    
        
        par = self.par
               
        self.q_in[ 0 ] = 0.0
        self.q_in[ 1 ] = 0.0
        self.q_in[ 2 ] =  par[ 1 ][ 4 ] 
        self.q_in[ 3 ] = -par[ 1 ][ 5 ] * par[ 1 ][ 6 ] / par[ 1 ][ 2 ] * 1.0 # Make 0.0 instead of 1.0 to turn off that parameter
            
        return

    #---------------------------------------
    def ode_sys( self, t, q ):  # implement the differential equations in this function
                                # parameters may be specified through the self.par[][] array
        
        #-----------------------------------------------------
        
        dq_dt = numpy.zeros(  self.numb_of_eq, dtype = float )
        par = self.par
        
        #-----------------------------------------------------
       
        dq_dt[ 0 ] = q[ 1 ]
        
        dq_dt[ 1 ] = - par[ 1 ][ 0 ] * q[ 0 ] - par[ 1 ][ 1 ] * q[ 1 ] + par[ 1 ][ 2 ] * ( q[ 2 ] - par[ 1 ][ 4 ] ) + par[ 1 ][ 3 ] * q[ 3 ] + par[ 1 ][ 5 ] * math.sin( par[ 1 ][ 6 ] * t )

        dq_dt[ 2 ] = q[ 3 ]
        
        dq_dt[ 3 ] =   par[ 3 ][ 0 ] * q[ 0 ] + par[ 3 ][ 1 ] * q[ 1 ] - par[ 3 ][ 2 ] * ( q[ 2 ] - par[ 3 ][ 4 ] ) - par[ 3 ][ 3 ] * q[ 3 ]
                
        #-----------------------------------------------------
        
        print('-> ', end = '')
        
        return( dq_dt )

    #---------------------------------------
    def jacob( self, t, q ): # implement the differential-equation jacobian in this function 
                             # parameters may be specified through the self.par[][] array
        #-----------------------------------------------------
                             
        jac_mtrx = numpy.zeros( ( self.numb_of_eq, self.numb_of_eq ), dtype = float )
        par = self.par

        #-----------------------------------------------------

        #dq_dt[ 0 ] = q[ 1 ]
        
        jac_mtrx[ 0 ][ 0 ] = 0.0
        jac_mtrx[ 0 ][ 1 ] = 1.0   
        jac_mtrx[ 0 ][ 2 ] = 0.0
        jac_mtrx[ 0 ][ 3 ] = 0.0   

        #dq_dt[ 1 ] = - par[ 1 ][ 0 ] * q[ 0 ] - par[ 1 ][ 1 ] * q[ 1 ] + par[ 1 ][ 2 ] * ( q[ 2 ] - par[ 1 ][ 4 ] ) + par[ 1 ][ 3 ] * q[ 3 ] + par[ 1 ][ 5 ] * math.sin( par[ 1 ][ 6 ] * t )
        
        jac_mtrx[ 1 ][ 0 ] =  - par[ 1 ][ 0 ] 
        jac_mtrx[ 1 ][ 1 ] =  - par[ 1 ][ 1 ]
        jac_mtrx[ 1 ][ 2 ] =    par[ 1 ][ 2 ]
        jac_mtrx[ 1 ][ 3 ] =    par[ 1 ][ 3 ]   
       
        #dq_dt[ 2 ] = q[ 3 ]
        
        jac_mtrx[ 2 ][ 0 ] = 0.0
        jac_mtrx[ 2 ][ 1 ] = 0.0   
        jac_mtrx[ 2 ][ 2 ] = 0.0
        jac_mtrx[ 2 ][ 3 ] = 1.0   

        #dq_dt[ 3 ] =   par[ 3 ][ 0 ] * q[ 0 ] + par[ 3 ][ 1 ] * q[ 1 ] - par[ 3 ][ 2 ] * ( q[ 2 ] - par[ 3 ][ 4 ] ) - par[ 3 ][ 3 ] * q[ 3 ]
        
        jac_mtrx[ 3 ][ 0 ] =    par[ 3 ][ 0 ] 
        jac_mtrx[ 3 ][ 1 ] =    par[ 3 ][ 1 ]
        jac_mtrx[ 3 ][ 2 ] =  - par[ 3 ][ 2 ]
        jac_mtrx[ 3 ][ 3 ] =  - par[ 3 ][ 3 ]

        #-----------------------------------------------------
        
        return( jac_mtrx )

    #---------------------------------------

#-------------------------------------------------------
#-------------------------------------------------------
class Solve_Differential_System( object ):

    #---------------------------------------
    def __init__( self ):
        
        self.comp_sol = []
         
        return
    #---------------------------------------
    def solve_sys( self, diff_sys ):

        time_range, time = diff_sys.time_set()
        
        q_in = diff_sys.init_cond()
 
        print('...working...')
        self.comp_sol = solve_ivp( diff_sys.ode_sys, time_range, q_in, method = 'Radau', t_eval = time, dense_output = True, atol = 1.0e-9, rtol = 1.0e-9, jac = diff_sys.jacob )
        
        self.data_matrix()
       
        return
    
    #---------------------------------------
    def data_matrix( self ):    

        self.data = numpy.zeros( ( len( self.comp_sol.t) , len( self.comp_sol.y ) + 1 ), dtype = float )
        
        number_of_rows = len( self.data )
        
        for i in range( number_of_rows ):
            self.data[ i ][ 0 ] = self.comp_sol.t[ i ]
            j = 1
            for var in ( self.comp_sol.y ):
                self.data[ i ][ j ] = var[ i ]
                j = j + 1
                
        return

    #---------------------------------------
    def print_to_file( self ):
        
        x = self.comp_sol
        
        fs = open( 'solution.txt', 'wt')
        
        number_of_rows      = len( x.t )

        for i in range( number_of_rows ):
            string = str( x.t[ i ] ) + ' '  
            for y in ( x.y ):
                string = string + str( y[ i ] ) + ' '  

            fs.write( string )
      
            fs.write( '\n' )

        fs.close()

        return
    
    #---------------------------------------
    def print_to_screen( self ):

        x = self.comp_sol
   
        number_of_rows      = len( x.t )
        print()
        
       
        for i in range( number_of_rows ):
            string = str( x.t[ i ] ) + ' '  
            for y in ( x.y ):
                string = string + str( y[ i ] ) + ' '  
	
            print( string )

        return
#-------------------------------------------------------
#-------------------------------------------------------- 
class Plot( object ):
       
    #----------------------------------------------------
    def __init__( self ):
        
        import matplotlib.pyplot as plt
        
        self.plt = plt
    
        self.data = []  
        
        return
    #---------------------------------------------------- 
    #----------------------------------------------------    
    def chart_it( self, data ):
    
        self.i_plot = True 
        
        while ( self.i_plot == True ):
            
            self.chart_it_now( data )
            
            #i_plot = int( input( "Enter '1' for next plot or '0' to exit: ") )
        
        return
    #---------------------------------------------------- 
    #----------------------------------------------------    
    def chart_it_now( self, data ):

        number_of_rows    = len( data )
        number_of_columns = len( data[ 0 ] )
            
        #-------------------------------------------       
        def function_to_plot():
                
            i   = number_of_columns - 1
                
            if ( i > 1 ):
                
                i_msg_1 = "There are " + str( i ) + " functions in this data set \n"
                i_msg_2 = "Enter either '0' to exit or \n"
                i_msg_3 = "Enter the number of the function to plot, between 1 and " + str(i) + ": "
                i_msg   = i_msg_1 + i_msg_2 + i_msg_3             
                
            else:
                
                i_msg_1 = "There is one function in this data set \n"
                i_msg_2 = "Enter either '0' to exit or \n"
                i_msg_3 = "Enter '1' to plot the function: "
                i_msg   = i_msg_1 + i_msg_2 + i_msg_3
                
            print()
            self.ifunct = int( input( i_msg  ) )
            print()
                
            if ( self.ifunct == 0 ):
                self.i_plot = False
                
            return
        #-------------------------------------------
        #-------------------------------------------
        def data_preparation():
                
            for j in range( number_of_columns ):
                row = []
                
                for i in range( number_of_rows ):
                    row.append( data[ i ][ j ] )
            
                self.data.append( row )
                                
            self.number_of_functions = number_of_columns
            
            return
        #-------------------------------------------
            
        function_to_plot()
        
        if ( self.i_plot == True ):
            data_preparation()
           
            self.init_plot()
            self.plot_it()
            
        return
    
    #----------------------------------------------------
    #----------------------------------------------------
    def init_plot( self ):
        
        # https://matplotlib.org/stable/api/markers_api.html
       
        """
        ls = '-' 	solid    line style
        ls = '--' 	dashed   line style
        ls = '-.' 	dash-dot line style
        ls = ':' 	dotted   line style
        """
        
        """
        'r' 	Red 	
        'g' 	Green 	
        'b' 	Blue 	
        'c' 	Cyan 	
        'm' 	Magenta 	
        'y' 	Yellow 	
        'k' 	Black 	
        'w' 	White
        """
        
        self.fig, self.ax = self.plt.subplots()
        
        self.ax.grid( True )
                
        self.ax.plot( self.data[ 0 ], self.data[ self.ifunct ],  ls = "-", linewidth = 1, color = 'g', marker = 'o', ms = 4, mec = 'b', mfc = 'b'  )
        
        return
    #----------------------------------------------------         
    #----------------------------------------------------  
    def plot_it( self ):
        
        self.plt.show()
        
        return
    #---------------------------------------------------- 
       
#-------------------------------------------------------
#-------------------------------------------------------
class Define_Points:
    
    #---------------------------------------
    def __init__( self, number_of_moving_bodies, sol ):
                
        self.number_of_moving_points  = number_of_moving_bodies
               
        self.sol = sol
        
        self.number_of_rows    = len( self.sol.t )
        self.number_of_columns = len( self.sol.y )
        
        self.x = numpy.zeros( ( self.number_of_moving_points + 1, self.number_of_rows ), dtype = float )
        self.y = numpy.zeros( ( self.number_of_moving_points + 1, self.number_of_rows ), dtype = float )  
        
          
        return
    #---------------------------------------
    #---------------------------------------
    def point_coordinates( self ):
        
        x = self.x
        y = self.y
           
        #--------------------------------------- 
        
        for j in range( self.number_of_moving_points ):
                 
            for i in range( self.number_of_rows ):
                #coordinates of moving points
            
                x[ j + 1 ][ i ] = self.sol.y[ 2*j ][ i ]
                y[ j + 1 ][ i ] = 0.0
            
        #--------------------------------------- 
        
        self.x = x
        self.y = y
       
        return
    #---------------------------------------
    
#-------------------------------------------------------
#-------------------------------------------------------
class Perform_Animation:
    
    #------------------------------------    
    def __init__( self, delta_t, coord ):
        

        self.plt = plt
        
        self.number_of_moving_points = coord.number_of_moving_points
        
        self.number_of_rows = coord.number_of_rows
        
        self.line = []

        for k in range( 2 * self.number_of_moving_points ):
            
            self.line.append( [] )
        
        self.interval    = 0.01
        #self.history_len = int( self.number_of_rows / 5 ) # how many trajectory points to keep
        
        self.fg_x = 5
               
        self.t_a = 0.05
        self.t_b = 0.95  

        self.dt =  delta_t
        
        self.x = coord.x
        self.y = coord.y
        
        return
    #------------------------------------    
    #------------------------------------    
    def set_up_animation( self ):        
    
        
        self.fig, self.ax = self.plt.subplots()
        
        ax = self.ax

        ax.grid( True )
        
       
        for k in range( self.number_of_moving_points ):           
            self.line[ k ], = ax.plot( [], [], ls = "-", linewidth = 0, color = 'g', marker = 's', ms = 80, mec = 'b', mfc = 'b' )
            

        for k in range( self.number_of_moving_points ):           
            self.line[ k + self.number_of_moving_points ], = ax.plot( [], [], ls = "--", linewidth = 4, color = 'k', marker = '8', ms = 8, mec = 'g', mfc = 'g' )
            

        self.time_template = 'time = %.2f' 
        self.time_text = ax.text( self.t_a, self.t_b, '', transform = ax.transAxes )
        
        self.ax = ax 
        
        #self.ax.set_xlim( [ min( self.x[ 1 ] ), max( self.x[ 1 ] ) ] )  
        #self.ax.set_ylim( [ min( self.y[ 1 ] ), max( self.y[ 1 ] ) ] )      
              
        #self.ax.plot( [ 0.0 ], [ 0.0 ], ls = "-", linewidth = 1, color = 'y', marker = 'o', ms = 20, mec = 'r', mfc = 'y' )
        
        for j in range( self.number_of_moving_points ):
            self.ax.plot( self.x[ j + 1 ], self.y[ j + 1 ], ls = "-", linewidth = 1, color = 'r', marker = ',', ms = 0, mec = 'r', mfc = 'r' )

        self.animate()
 
        return
    #------------------------------------    
    #------------------------------------    
    def animate( self ):
        
        
        ani = animation.FuncAnimation( self.fig, self.show, self.number_of_rows, interval = self.interval * self.dt, blit = True )
        plt.show()  
        
        print( ani )
    
        return
    #------------------------------------    
    #------------------------------------    
    def show( self, j ):
        
        i = 1 * j
        if ( i > self.number_of_rows ):
            i = self.number_of_rows - 1

        var = []    
        

        for k in range( self.number_of_moving_points ):
            self.line[ k ].set_data( [ self.x[ k + 1 ][ i ] ], [ self.y[ k + 1 ][ i ] ] )
            var.append( self.line[ k ] )


        self.line[ self.number_of_moving_points ].set_data( [ -0.5, self.x[ 1 ][ i ] ], [ self.y[ 1 ][ i ], self.y[ 1 ][ i ] ] )       
        var.append(  self.line[ self.number_of_moving_points ] )


        for k in range( 1, self.number_of_moving_points ):
            self.line[ self.number_of_moving_points + k ].set_data( [ self.x[ k ][ i ], self.x[ k + 1 ][ i ] ], [ self.y[ k ][ i ], self.y[ k + 1 ][ i ] ] )
            var.append( self.line[ self.number_of_moving_points + k ] )
            
        
        self.time_text.set_text( self.time_template % ( i * self.dt ) )
        
        var.append( self.time_text )
        
        return( var )
    
    #------------------------------------ 
#-------------------------------------------------------
#-------------------------------------------------------
class Eigenvalue_Sets:

    #---------------------------------------
    def __init__( self ):

        self.define_eigenvalue_parameters( )
        self.eigen_par = []
        self.all_eigenvalues = []
        self.stacked_eigenvalues = []
    #---------------------------------------
    #---------------------------------------
    def define_eigenvalue_parameters( self ):
        
        print()
        self.choice      =    str( input( 'Is an eigenvalue analysis required? Enter "yes" or "no": ' ) )
        print()
        
        if ( self.choice == 'yes' ) :        
        
            print( 'Definition of limits of eigenvalue parameter')
            self.eig_par_min     =  float( input( 'Enter lower limit: ' ) )
            self.eig_par_max     =  float( input( 'Enter upper limit: ' ) )
            if ( self.eig_par_max <= self.eig_par_min ):
                self.eig_par_max = 1.0 + 0.1 * self.eig_par_min
        
            self.number_of_eig_sets =  int( input( 'Enter desired total numer of eigenvalue sets: ' ) )
        
            if( self.number_of_eig_sets < 1 ):
                self.number_of_eig_sets = 1

        return
    #---------------------------------------
    #---------------------------------------
    def calculate_eigenvalues( self, jacobian ):

        q = numpy.zeros( jacobian.numb_of_eq, dtype = float ) 

        eigenvalues = []
         
        print('...working...')
        
        eig_par_ratio = ( self.eig_par_max - self.eig_par_min ) / float( self.number_of_eig_sets - 1 )
        
        for i in range( self.number_of_eig_sets   ):
            print( '->', end = '' )
            
            eig_par = self.eig_par_min + float( i ) * eig_par_ratio
            
            jacobian.eig_par = eig_par
            jacobian.number_of_equations_and_parameters_2()
            dq_p_dq = jacobian.jacob( 0.0, q )
            
            eigens = numpy.linalg.eigvals( dq_p_dq )
            
            for j in range ( jacobian.numb_of_eq ):
                eigs = [] 

                eigs.append( numpy.real( eigens[ j ] ) )
                eigs.append( numpy.imag( eigens[ j ] )  )
                eigenvalues.append( eigs )

                      
            self.all_eigenvalues.append( eigenvalues )
            eigenvalues = []
            
            eigs = []
            eigs.append( i + 1 )
            eigs.append( eig_par )
            self.eigen_par.append( eigs )
            eigs = []
            
        self.sort_eigenvalues()
        self.stack_eigenvalues()
        
        print()
        
        return
    #---------------------------------------
    #---------------------------------------
    def sort_eigenvalues( self ):

        eps = 1.0e-09

        number_of_rows    = len( self.eigen_par )
        
        number_of_columns = len( self.all_eigenvalues[ 0 ] )
        
        for j in range( number_of_columns - 1 ):

            a_r   = self.all_eigenvalues[ 0 ][ j ][ 0 ] 
            a_i   = self.all_eigenvalues[ 0 ][ j ][ 1 ]
            b_r   = a_r
            b_i   = a_i
            k_min = j
            
            for k in range( j + 1, number_of_columns ):
                c_i = self.all_eigenvalues[ 0 ][ k ][ 1 ]
                
                if ( abs( c_i ) > abs( a_i ) ):
                    a_i     = c_i
                    k_min   = k
                    
            
            if ( k_min > j ):
                self.all_eigenvalues[ 0 ][ j ][ 0 ]  = self.all_eigenvalues[ 0 ][ k_min ][ 0 ]
                self.all_eigenvalues[ 0 ][ j ][ 1 ]  = self.all_eigenvalues[ 0 ][ k_min ][ 1 ]
                
                self.all_eigenvalues[ 0 ][ k_min ][ 0 ]  = b_r
                self.all_eigenvalues[ 0 ][ k_min ][ 1 ]  = b_i


        for j in range( number_of_columns - 1 ):
            
                        
            for i in range( 1, number_of_rows ):
                
                a_r = self.all_eigenvalues[ i - 1 ][ j ][ 0 ]
                a_i = self.all_eigenvalues[ i - 1 ][ j ][ 1 ]
                
                if ( abs( a_i ) < eps ):
                    selection = 2
                else :
                    selection = 1
                                              
                b_r = self.all_eigenvalues[ i ][ j ][ 0 ]
                b_i = self.all_eigenvalues[ i ][ j ][ 1 ]
            
                d_r = abs( b_r - a_r )
                d_i = abs( b_i - a_i )                
                k_min = j
                
                for k in range( j + 1, number_of_columns ):
                    c_r = self.all_eigenvalues[ i ][ k ][ 0 ]
                    c_i = self.all_eigenvalues[ i ][ k ][ 1 ]
                    
                    if   ( selection == 1 and ( abs( c_i - a_i ) < d_i ) ):
                        d_i     = abs( c_i - a_i )
                        k_min   = k
                        
                    elif ( selection == 2 and ( abs( c_r - a_r ) < d_r ) ):
                        d_r     = abs( c_r - a_r )
                        k_min   = k

                    
                self.all_eigenvalues[ i ][ j ][ 0 ]  = self.all_eigenvalues[ i ][ k_min ][ 0 ]
                self.all_eigenvalues[ i ][ j ][ 1 ]  = self.all_eigenvalues[ i ][ k_min ][ 1 ]
                
                self.all_eigenvalues[ i ][ k_min ][ 0 ]  = b_r
                self.all_eigenvalues[ i ][ k_min ][ 1 ]  = b_i
                     
        return
    #---------------------------------------
    #---------------------------------------
    def stack_eigenvalues( self ):

        eigs = []
        
        for i in range( self.number_of_eig_sets   ):

            
            a     = self.all_eigenvalues[ i ][ 0 ][ 0 ]
            j_min = 0
            
            number_of_columns = len( self.all_eigenvalues[ i ]  )
            
            for j in range ( 1, number_of_columns ):
                
                if ( self.all_eigenvalues[ i ][ j ][ 0 ] > a ) :
                    a     = self.all_eigenvalues[ i ][ j ][ 0 ]
                    j_min = j
                
            for j in range ( number_of_columns ):
                
                eigs.append( self.eigen_par[ i ][ 1 ] )
                eigs.append( self.all_eigenvalues[ i ][ j ][ 0 ] )
                eigs.append( self.all_eigenvalues[ i ][ j ][ 1 ] )
                
                eigs.append( self.all_eigenvalues[ i ][ j_min ][ 0 ] )
                eigs.append( self.all_eigenvalues[ i ][ j_min ][ 1 ] )

                self.stacked_eigenvalues.append( eigs )   
                eigs = [] 
               
        print()
        
        return
    #---------------------------------------
    #---------------------------------------
    def print_to_screen( self ):
        
        number_of_rows = len( self.eigen_par )
        
        for i in range( number_of_rows ):
            print( self.eigen_par[ i ][ 0 ], ' ', self.eigen_par[ i ][ 1 ], end = '')
            
            number_of_columns = len( self.all_eigenvalues[ i ]  ) 
            
            for j in range( number_of_columns ):
                
                print( ' ', self.all_eigenvalues[ i ][ j ][ 0 ], ' ', self.all_eigenvalues[ i ][ j ][ 1 ],  end = '')    
        
            print()
            
            
        number_of_rows = len( self.stacked_eigenvalues )
        
        for i in range( number_of_rows ):
            
            number_of_columns = len( self.stacked_eigenvalues[ i ]  ) 
            
            for j in range( number_of_columns ):
                
                print( ' ', self.stacked_eigenvalues[ i ][ j ],  end = '')    
        
            print()
        
        
        return
    #---------------------------------------
    #---------------------------------------
    def print_to_file( self ):
                
        fs = open( 'sol_eigenvalues.txt', 'wt')
        number_of_rows = len( self.eigen_par )

        for i in range( number_of_rows ):
            string = str( self.eigen_par[ i ][ 0 ] ) + ' ' + str( self.eigen_par[ i ][ 1 ] )
            
            number_of_columns = len( self.all_eigenvalues[ i ]  )  

            for j in range( number_of_columns ):
                string = string + ' ' + str( self.all_eigenvalues[ i ][ j ][ 0 ] ) + ' ' +  str( self.all_eigenvalues[ i ][ j ][ 1 ] ) 

            fs.write( string )

            fs.write( '\n' )

        fs.close()
        

        fs = open( 'stacked_eigenvalues.txt', 'wt')


        number_of_rows = len( self.stacked_eigenvalues )
        
        for i in range( number_of_rows ):
            
            number_of_columns = len( self.stacked_eigenvalues[ i ]  )
            
            string = ''
            
            for j in range( number_of_columns - 1 ):
                
                string = string + str( self.stacked_eigenvalues[ i ][ j ] ) + ' '
                
                if ( j == 1 ): 
                    string = string + str( self.stacked_eigenvalues[ i ][ 0 ] ) + ' '   
                
            j      = number_of_columns - 1
            string = string + str( self.stacked_eigenvalues[ i ][ j ] )
                
            fs.write( string )
            
            fs.write( '\n' )
            
                
        fs.close()
        
        return
    #------------------------------------ 

#-------------------------------------------------------
main()
#-------------------------------------------------------










