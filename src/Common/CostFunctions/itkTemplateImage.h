#ifndef __itkTemplateImage_H__
#define __itkTemplateImage_H__

#include "itkObject.h"

namespace itk
{
    
    class TemplateImage : public Object
    {
    public:
        
        typedef TemplateImage                          Self;
        typedef Object                     Superclass;
        typedef SmartPointer< Self >                                    Pointer;
        typedef SmartPointer< const Self >                              ConstPointer;
        
        itkTypeMacro( TemplateImage, Object );

        virtual void Initialize( void ) throw ( ExceptionObject );
        
        TemplateImage();
        
        virtual ~TemplateImage();
        virtual double DetermineHistogramMin(const std::vector<double> & imageValues, const unsigned int n) const = 0;
        virtual double DetermineHistogramMax(const std::vector<double> & imageValues, const unsigned int n) const = 0;

        virtual double CalculateIntensity(const std::vector<double> & imageValues, const unsigned int n, const std::vector<unsigned int> & positions, const unsigned int o) const = 0;
        virtual double CalculateRatio(const double & imageValue, const double & templateIntensity, const unsigned int position, const unsigned int ref, const unsigned int total) const = 0;
        
        inline void SetIntensityConstants(std::vector<double> & IC){ m_IntensityConstants = IC;}
        virtual double TransformIntensity(const double & imageValue, const unsigned int n) const =0;
        
    protected:
        
        std::vector<double> m_IntensityConstants;
        
    private:
        
        TemplateImage( const Self & ); // purposely not implemented
        void operator=( const Self & );

    };
    
}

#ifndef ITK_MANUAL_INSTANTIATION
#endif

#endif // end #ifndef __itkTemplateImage_H__
