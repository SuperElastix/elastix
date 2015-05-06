#ifndef __itkTemplateImage_H__
#define __itkTemplateImage_H__

namespace itk
{
    
    class TemplateImage : public DataObject
    {
    public:
        
        typedef TemplateImage                          Self;
        typedef SmartPointer< Self >                                    Pointer;
        typedef SmartPointer< const Self >                              ConstPointer;
                
        virtual void Initialize( void ) throw ( ExceptionObject );
        
        TemplateImage();
        
        virtual ~TemplateImage();
        
        virtual double CalculateIntensity(const std::vector<double> & imageValues, const unsigned int n) const = 0;
        virtual double CalculateIntensity(const std::vector<double> & imageValues, const unsigned int n, const std::vector<unsigned int> & positions) const = 0;
        virtual double CalculateRatio(const double & imageValue, const double & templateIntensity, const unsigned int position) const = 0;
        
        inline void SetIntensityConstants(std::vector<double> & IC){ m_IntensityConstants = IC;}
        
    protected:
        
        std::vector<double> m_IntensityConstants;
        
    private:
        
        TemplateImage( const Self & ); // purposely not implemented
        void operator=( const Self & );

    };
    
}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkTemplateImage.hxx"
#endif

#endif // end #ifndef __itkTemplateImage_H__
