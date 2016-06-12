#ifndef __itkMetricCombiner_H__
#define __itkMetricCombiner_H__

#include "itkDataObject.h"

#include "itkArray.h"

namespace itk
{
    class MetricCombiner : public DataObject
    {
    public:
        
        typedef MetricCombiner                          Self;
        typedef SmartPointer< Self >                                    Pointer;
        typedef SmartPointer< const Self >                              ConstPointer;
                
        virtual void Initialize( void ) throw ( ExceptionObject );
        
        MetricCombiner();
        
        virtual ~MetricCombiner();
        
        virtual void Combine(double & value, std::vector<double> & values, unsigned int n, const bool minimize) const =0;

        virtual void Combine(double & value, Array<double> & derivative, std::vector<double> & values, std::vector< Array<double> > & derivatives, unsigned int n, const bool minimize) const =0;
        
    protected:
        
    private:
        
        MetricCombiner( const Self & ); // purposely not implemented
        void operator=( const Self & );

    };
    
}

#ifndef ITK_MANUAL_INSTANTIATION
#endif

#endif // end #ifndef __itkMetricCombiner_H__
