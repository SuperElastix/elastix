#ifndef __itkScaledSingleValuedCostFunction_h
#define __itkScaledSingleValuedCostFunction_h

#include "itkSingleValuedCostFunction.h"

namespace itk
{
  /** 
   * \class ScaledSingleValuedCostFunction
   * \brief A cost function that applies a scaling to another cost function
   *
   * This class can be used to adapt an existing, badly scaled, cost function. 
   * 
   * By default it does not apply any scaling. Use the method SetUseScales(true)
   * to enable the use of scales.
   *
   * \ingroup Numerics
   */

  class ScaledSingleValuedCostFunction : public SingleValuedCostFunction
  {
  public:
    typedef ScaledSingleValuedCostFunction  Self;
    typedef SingleValuedCostFunction        Superclass;
    typedef SmartPointer<Self>              Pointer;
    typedef SmartPointer<const Self>        ConstPointer;

    itkNewMacro(Self);
    itkTypeMacro(ScaledSingleValuedCostFunction, SingleValuedCostFunction);

    typedef Superclass::MeasureType         MeasureType;
    typedef Superclass::DerivativeType      DerivativeType;
    typedef Superclass::ParametersType      ParametersType;

    typedef Superclass::Pointer             SingleValuedCostFunctionPointer;
    typedef Array<double>                   ScalesType;

    /** Divide the parameters by the scales and call the GetValue routine 
     * of the unscaled cost function */
    virtual MeasureType GetValue(
      const ParametersType & parameters ) const;

    /** Divide the parameters by the scales, call the GetDerivative routine
     * of the unscaled cost function and divide the resulting derivative by
     * the scales */
    virtual void GetDerivative(
      const ParametersType & parameters,
      DerivativeType & derivative ) const;

    /** Same procedure as in GetValue and GetDerivative */
    virtual void GetValueAndDerivative(
      const ParametersType & parameters,
      MeasureType & value,
      DerivativeType & derivative ) const;

    /** Ask the UnscaledCostFunction how many parameters it has. */
    virtual unsigned int GetNumberOfParameters(void) const;

    /** Set/Get the cost function that needs scaling */
    itkSetObjectMacro(UnscaledCostFunction, Superclass);
    itkGetObjectMacro(UnscaledCostFunction, Superclass);

    /** Set/Get the scales */
    virtual void SetScales(const ScalesType & scales);
    itkGetConstReferenceMacro(Scales, ScalesType);

    /** Select whether to use scales or not */
    itkSetMacro(UseScales, bool);
    itkGetConstMacro(UseScales, bool);

    itkBooleanMacro(NegateCostFunction);
    itkSetMacro(NegateCostFunction, bool);
    itkGetConstMacro(NegateCostFunction, bool);

    /** x = y/s  */
    virtual void ConvertScaledToUnscaledParameters(ParametersType & parameters) const;

    /** y = x*s  */
    virtual void ConvertUnscaledToScaledParameters(ParametersType & parameters) const;

  protected:
    ScaledSingleValuedCostFunction();
    virtual ~ScaledSingleValuedCostFunction() {};
    void PrintSelf(std::ostream& os, Indent indent) const{};


  private:
    ScaledSingleValuedCostFunction(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented

    ScalesType                            m_Scales;
    SingleValuedCostFunctionPointer       m_UnscaledCostFunction;
    bool                                  m_UseScales;      
    bool                                  m_NegateCostFunction;
    

  }; // end class ScaledSingleValuedCostFunction

  

} //end namespace itk


#endif // #ifndef __itkScaledSingleValuedCostFunction_h

