#ifndef __itkMetricSquaredSum_H__
#define __itkMetricSquaredSum_H__

#include "itkMetricCombiner.h"

namespace itk
{
    
    class MetricSquaredSum : public MetricCombiner
    {
    public:
        
        typedef MetricSquaredSum                          Self;
        typedef MetricCombiner                          Superclass;
        typedef SmartPointer< Self >                                    Pointer;
        typedef SmartPointer< const Self >                              ConstPointer;
        
        
        itkTypeMacro( MetricSquaredSum, MetricCombiner );
        
        virtual void Initialize( void ) throw ( ExceptionObject );
        
        MetricSquaredSum();
        
        virtual ~MetricSquaredSum();
        
        virtual void Combine(double & value, std::vector<double> & values, unsigned int n, const bool minimize) const;
        
        virtual void Combine(double & value, Array<double> & derivative, std::vector<double> & values, std::vector< Array<double> > & derivatives, unsigned int n,const bool minimize) const;
        
    protected:
        
    private:
        
        MetricSquaredSum( const Self & ); // purposely not implemented
        void operator=( const Self & );

    };
    
}

#ifndef ITK_MANUAL_INSTANTIATION
#endif

#endif // end #ifndef __itkMetricSquaredSum_H__
