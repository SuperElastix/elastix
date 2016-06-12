#ifndef __itkMetricSum_H__
#define __itkMetricSum_H__

#include "itkMetricCombiner.h"

namespace itk
{
    
    class MetricSum : public MetricCombiner
    {
    public:
        
        typedef MetricSum                          Self;
        typedef MetricCombiner Superclass;
        typedef SmartPointer< Self >                                    Pointer;
        typedef SmartPointer< const Self >                              ConstPointer;
                
        itkTypeMacro( MetricSum, MetricCombiner );
        
        virtual void Initialize( void ) throw ( ExceptionObject );
        
        MetricSum();
        
        virtual ~MetricSum();
        
        virtual void Combine(double & value, std::vector<double> & values, unsigned int n, const bool minimize) const;
        
        virtual void Combine(double & value, Array<double> & derivative, std::vector<double> & values, std::vector< Array<double> > & derivatives, unsigned int n, const bool minimize) const;
        
    protected:
        
    private:
        
        MetricSum( const Self & ); // purposely not implemented
        void operator=( const Self & );

    };
    
}

#ifndef ITK_MANUAL_INSTANTIATION
#endif

#endif // end #ifndef __itkMetricSum_H__
