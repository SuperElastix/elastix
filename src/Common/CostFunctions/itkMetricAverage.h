#ifndef __itkMetricAverage_H__
#define __itkMetricAverage_H__

#include "itkMetricCombiner.h"

namespace itk
{
    
    class MetricAverage : public MetricCombiner
    {
    public:
        
        typedef MetricAverage                          Self;
        typedef MetricCombiner Superclass;
        typedef SmartPointer< Self >                                    Pointer;
        typedef SmartPointer< const Self >                              ConstPointer;
                
        itkTypeMacro( MetricAverage, MetricCombiner );
        
        virtual void Initialize( void ) throw ( ExceptionObject );
        
        MetricAverage();
        
        virtual ~MetricAverage();
        
        virtual void Combine(double & value, std::vector<double> & values, unsigned int n, const bool minimize) const;
        
        virtual void Combine(double & value, Array<double> & derivative, std::vector<double> & values, std::vector< Array<double> > & derivatives, unsigned int n, const bool minimize) const;
        
    protected:
        
    private:
        
        MetricAverage( const Self & ); // purposely not implemented
        void operator=( const Self & );

    };
    
}

#ifndef ITK_MANUAL_INSTANTIATION
#endif

#endif // end #ifndef __itkMetricAverage_H__
