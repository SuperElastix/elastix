#ifndef __itkAATemplate_H__
#define __itkAATemplate_H__

#include "itkTemplateImage.h"

namespace itk
{
    class AATemplate : public TemplateImage
    {
    public:
        
        typedef AATemplate                          Self;
        typedef TemplateImage Superclass;
        typedef SmartPointer< Self >                                    Pointer;
        typedef SmartPointer< const Self >                              ConstPointer;
        
        itkTypeMacro( AATemplate, TemplateImage );
        
        virtual void Initialize( void ) throw ( ExceptionObject );
        
        AATemplate();
        
        virtual ~AATemplate();
        
        virtual double DetermineHistogramMin(const std::vector<double> & imageValues, const unsigned int n) const;
        virtual double DetermineHistogramMax(const std::vector<double> & imageValues, const unsigned int n) const;
        virtual double CalculateIntensity(const std::vector<double> & imageValues, const unsigned int n, const std::vector<unsigned int> & positions, const unsigned int o) const;

        virtual double CalculateRatio(const double & imageValue, const double & templateIntensity, const unsigned int position, const unsigned int ref, const unsigned int total) const;
        inline virtual double TransformIntensity(const double & imageValue, const unsigned int n) const {return imageValue;}


    protected:
        
    private:
        
        AATemplate( const Self & ); // purposely not implemented
        void operator=( const Self & );

    };
    
}

#ifndef ITK_MANUAL_INSTANTIATION
#endif

#endif // end #ifndef __itkAATemplate_H__
