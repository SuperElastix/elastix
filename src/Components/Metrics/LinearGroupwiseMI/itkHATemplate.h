#ifndef __itkHATemplate_H__
#define __itkHATemplate_H__

#include "itkTemplateImage.h"

namespace itk
{
    
    class HATemplate : public TemplateImage
    {
    public:
        
        typedef HATemplate                          Self;
        typedef TemplateImage Superclass;
        typedef SmartPointer< Self >                                    Pointer;
        typedef SmartPointer< const Self >                              ConstPointer;
        
        itkNewMacro( Self) ;
        
        itkTypeMacro( HATemplate, TemplateImage );
        
        virtual void Initialize( void ) throw ( ExceptionObject );
        
        HATemplate();
        
        virtual ~HATemplate();
        
        virtual double CalculateIntensity(const std::vector<double> & imageValues, const unsigned int n) const;
        virtual double CalculateIntensity(const std::vector<double> & imageValues, const unsigned int n, const std::vector<unsigned int> & positions) const;
        virtual double CalculateRatio(const double & imageValue, const double & templateIntensity, const unsigned int position) const;

        
    protected:
        
    private:
        
        HATemplate( const Self & ); // purposely not implemented
        void operator=( const Self & );

    };
    
}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkHATemplate.hxx"
#endif

#endif // end #ifndef __itkHATemplate_H__
