#ifndef __itkRSGDEachParameterApartOptimizer_h
#define __itkRSGDEachParameterApartOptimizer_h

#include "itkRSGDEachParameterApartBaseOptimizer.h"

namespace itk
{
  
/** \class RSGDEachParameterApartOptimizer
 * \brief Implement a gradient descent optimizer
 *
 *
 */
class RSGDEachParameterApartOptimizer : 
    public RSGDEachParameterApartBaseOptimizer
{
public:
  /** Standard class typedefs. */
  typedef RSGDEachParameterApartOptimizer         Self;
  typedef RSGDEachParameterApartBaseOptimizer     Superclass;
  typedef SmartPointer<Self>                          Pointer;
  typedef SmartPointer<const Self>                    ConstPointer;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);
  
  /** Run-time type information (and related methods). */
  itkTypeMacro( RSGDEachParameterApartOptimizer, 
                RSGDEachParameterApartBaseOptimizer );

  /** Cost function typedefs. */
  typedef Superclass::CostFunctionType        CostFunctionType;
  typedef CostFunctionType::Pointer           CostFunctionPointer;
  

protected:
  RSGDEachParameterApartOptimizer() {};
  virtual ~RSGDEachParameterApartOptimizer() {};

  /** Advance one step along the corrected gradient taking into
   * account the steplengths represented by the factor array.
   * This method is invoked by AdvanceOneStep. It is expected
   * to be overrided by optimization methods in non-vector spaces
   * \sa AdvanceOneStep */
  virtual void StepAlongGradient( 
    const DerivativeType & factor, 
    const DerivativeType & transformedGradient );

private:
  RSGDEachParameterApartOptimizer(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace itk



#endif



