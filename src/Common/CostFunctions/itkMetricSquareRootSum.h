#ifndef __itkMetricSquareRootSum_H__
#define __itkMetricSquareRootSum_H__

#include "itkMetricCombiner.h"

namespace itk
{
    
    class MetricSquareRootSum : public MetricCombiner
    {
    public:
        
        typedef MetricSquareRootSum                          Self;
        typedef MetricCombiner                          Superclass;
        typedef SmartPointer< Self >                                    Pointer;
        typedef SmartPointer< const Self >                              ConstPointer;
                
        itkTypeMacro( MetricSquareRootSum, MetricCombiner );
        
        virtual void Initialize( void ) throw ( ExceptionObject );
        
        MetricSquareRootSum();
        
        virtual ~MetricSquareRootSum();
        
        virtual void Combine(double & value, std::vector<double> & values, unsigned int n, const bool minimize) const;
        
        virtual void Combine(double & value, Array<double> & derivative, std::vector<double> & values, std::vector< Array<double> > & derivatives, unsigned int n,const bool minimize) const;
        
    protected:
        
    private:
        
        MetricSquareRootSum( const Self & ); // purposely not implemented
        void operator=( const Self & );

    };
    
}

#ifndef ITK_MANUAL_INSTANTIATION
#endif

#endif // end #ifndef __itkMetricSquareRootSum_H__
