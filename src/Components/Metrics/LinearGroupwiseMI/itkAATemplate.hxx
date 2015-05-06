#ifndef _itkAATemplate_HXX__
#define _itkAATemplate_HXX__

#include "itkAATemplate.h"

#ifdef ELASTIX_USE_OPENMP
#include <omp.h>
#endif

namespace itk
{
    /**
     * ********************* Constructor ******************************
     */
    
    AATemplate
    ::AATemplate()
    {
    }
    
    /**
     * ******************* Destructor *******************
     */
    
    AATemplate
    ::~AATemplate()
    {
    }
    
    /**
     * ******************* Initialize *******************
     */
    
    void
    AATemplate
    ::Initialize( void ) throw ( ExceptionObject )
    {
        this->Superclass::Initialize();
    }
    
    /**
     * ******************* CalculateIntensity *******************
     */
    
    double
    AATemplate
    ::CalculateIntensity(const std::vector<double> & imageValues, const unsigned int n) const
    {
        double templateIntensity =0;
        for(unsigned int i=0; i < n; i++)
        {
            templateIntensity += imageValues[i];
        }
        
        return templateIntensity / n;
        
    }
    
    /**
     * ******************* CalculateIntensity *******************
     */
    
    double
    AATemplate
    ::CalculateIntensity(const std::vector<double> & imageValues, const unsigned int n, const std::vector<unsigned int> & positions) const
    {
        double templateIntensity =0;
        for(unsigned int i=0; i < n; i++)
        {
            templateIntensity += imageValues[i];
        }
        
        return templateIntensity / n;
        
    }

    /**
     * ******************* CalculateRatio *******************
     */
    
    double
    AATemplate
    ::CalculateRatio(const double & imageValue, const double & templateIntensity, const unsigned int position) const
    {
        return 1.0;
    }


    
}

#endif // end #ifndef _itkAATemplate_HXX__
