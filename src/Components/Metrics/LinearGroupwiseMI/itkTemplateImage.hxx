#ifndef _itkTemplateImage_HXX__
#define _itkTemplateImage_HXX__

#include "itkTemplateImage.h"

#ifdef ELASTIX_USE_OPENMP
#include <omp.h>
#endif

namespace itk
{
    /**
     * ********************* Constructor ******************************
     */
    
    TemplateImage
    ::TemplateImage()
    {

    }
    
    /**
     * ******************* Destructor *******************
     */
    
    TemplateImage
    ::~TemplateImage()
    {
    }
    
    /**
     * ******************* Initialize *******************
     */
    
    void
    TemplateImage
    ::Initialize( void ) throw ( ExceptionObject )
    {
    }
        
}

#endif // end #ifndef _itkTemplateImage_HXX__
