from math import *
class Thresholds:
    def j_calc(m1, m2, x, y, z, xVel, yVel, zVel):

        #common representations (used to interpret the Jacobi Constant information website):
        #pi_2 = mass parameter
        #1-pi_2 = mass of body 1
        #sigma = dist between body 3 and body 1
        #psi = dist between body 3 and body 2
        
        #calculate mass parameter (represented as pi_2 in formula definitions)
        mP = m2/(m1+m2)

        #relative velocity of the tertiary mass, m3
        vel_tertiary = sqrt(xVel**2  + yVel**2)

        #distance between m3 and m1
        dist_m3m1 = sqrt((x + mP)**2 + (y)**2 + (z)**2)

        #distance between m3 and m2
        dist_m3m2 = sqrt((x - (1 - mP))**2 + (y)**2 + (z)**2)

        #calculate Jacobi Constant
        j_const = ((1/2)*(vel_tertiary)**2) - (m1/dist_m3m1) - (mP/dist_m3m2) - ((1/2)*((m1*(dist_m3m1**2)) + (mP*dist_m3m2**2)))

        return j_const
