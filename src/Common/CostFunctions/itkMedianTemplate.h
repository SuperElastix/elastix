#ifndef __itkMedianTemplate_H__
#define __itkMedianTemplate_H__

#include "itkTemplateImage.h"

namespace itk
{
    class MedianTemplate : public TemplateImage
    {
    public:
        
        typedef MedianTemplate                          Self;
        typedef TemplateImage Superclass;
        typedef SmartPointer< Self >                                    Pointer;
        typedef SmartPointer< const Self >                              ConstPointer;
                
        itkTypeMacro( MedianTemplate, TemplateImage );
        
        virtual void Initialize( void ) throw ( ExceptionObject );
        
        MedianTemplate();
        
        virtual ~MedianTemplate();
        
        virtual double DetermineHistogramMin(const std::vector<double> & imageValues, const unsigned int n) const;
        virtual double DetermineHistogramMax(const std::vector<double> & imageValues, const unsigned int n) const;
        virtual double CalculateIntensity(const std::vector<double> & imageValues, const unsigned int n, const std::vector<unsigned int> & positions, const unsigned int o) const;

        virtual double CalculateRatio(const double & imageValue, const double & templateIntensity, const unsigned int position, const unsigned int ref, const unsigned int total) const;
        inline virtual double TransformIntensity(const double & imageValue, const unsigned int n) const {return imageValue;}


    protected:
        
    private:
        
        MedianTemplate( const Self & ); // purposely not implemented
        void operator=( const Self & );

    };
    
}

#ifndef ITK_MANUAL_INSTANTIATION
#endif

#endif // end #ifndef __itkMedianTemplate_H__
