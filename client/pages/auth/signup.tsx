import * as React from "react";
import { Button, Stack, Typography, useTheme } from "@mui/material";
import { Box } from "@mui/system";
import Image from "next/image";
import { CustomLink, MuiInput } from "../../components/styled-components";
import Layout from "../../components/layout/secondary";
import { usePolybase } from "@polybase/react";
import { AppState } from "../_app";
import { useRouter } from "next/router";
import { Spinner } from "@chakra-ui/react";
import createNftUrl from "../../lib/utils/create-nft-url";
import sendConfirmMail from "../../lib/mail-service";

type FormStateType = {
  email: string;
  password: string;
  firstName: string;
  lastName: string;
  confPassword: string;
};

const SignUp: React.FC = () => {
  const router = useRouter();
  const polyDB = usePolybase();
  const theme = useTheme();
  const { state } = React.useContext(AppState);

  const initState: FormStateType = {
    email: "",
    password: "",
    firstName: "",
    lastName: "",
    confPassword: "",
  };

  const [formState, setFormState] = React.useState<FormStateType>(initState);
  const [isLoading, setIsLoading] = React.useState<boolean>(false);

  const updateFormState = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    setFormState((val) => ({
      ...val,
      [e.target.name]: e.target.value,
    }));
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    console.log(formState);
    const name = [formState.firstName, formState.lastName].join(" ");
    try {
      await polyDB
        .collection("UserSBT")
        .create([
          Date.now().toString(),
          name,
          state.address,
          Date.now(),
          Date.now(),
          "",
          [],
          "",
          "Daily",
          formState.email,
        ])
        .then(async (e) => {
          const metadataUri = createNftUrl(e.data.id);
          state.userSBT.methods
            .mint(state.address, name, metadataUri, e.data.id)
            .send({ from: state.address })
            .on("receipt", async (res) => {
              console.log(res.transactionHash);
              await sendConfirmMail(name, formState.email);
              await polyDB
                .collection("UserSBT")
                .record(e.data.id)
                .call("updateTxnHash", [res.transactionHash])
                .then(() =>
                  router.push("/interests").then(() => location.reload())
                );
            });
        });
      setIsLoading(false);
    } catch (e) {
      setIsLoading(false);
      console.log(e);
    }
  };

  return (
    <Stack direction={"row"}>
      <Stack
        flex={1}
        height={"100vh"}
        direction="column"
        alignItems={"start"}
        justifyContent={"start"}
      >
        <Box p="70px">
          <Image src="/images/logo.png" alt="Logo" height={30} width={150} />
        </Box>
        <Box mb="auto" mr="auto" ml={"auto"} mt={"10%"}>
          <Image
            src={"/images/signup.png"}
            alt="Login"
            width={1600 / 4}
            height={1600 / 4}
          />
        </Box>
      </Stack>
      <Box
        flex={1}
        display="flex"
        alignItems={"center"}
        justifyContent={"center"}
      >
        <Stack
          direction={"column"}
          width={"85%"}
          height={"80%"}
          justifyContent="start"
        >
          <Stack width={"100%"}>
            <Typography fontSize={"3em"}>Getting Started</Typography>
            <Typography fontSize={"0.9em"}>
              Already have an account?
              <CustomLink href="/auth/login"> Sign in</CustomLink>
            </Typography>
          </Stack>
          <form
            onSubmit={submit}
            style={{
              paddingTop: "50px",
              height: "100%",
            }}
          >
            <Stack height="100%" gap="50px">
              <Stack direction="row" width={"100%"} gap="50px">
                <MuiInput
                  value={formState.firstName}
                  label={"First name"}
                  name={"firstName"}
                  onChange={updateFormState}
                />
                <MuiInput
                  value={formState.lastName}
                  label={"Last name"}
                  name={"lastName"}
                  onChange={updateFormState}
                />
              </Stack>
              <MuiInput
                label="Email"
                value={formState.email}
                onChange={updateFormState}
              />
              <MuiInput
                password
                label="Password"
                value={formState.password}
                onChange={updateFormState}
              />
              <MuiInput
                password
                label="Confirm password"
                value={formState.confPassword}
                onChange={updateFormState}
                name={"confPassword"}
              />

              <Button
                disabled={isLoading}
                sx={{
                  alignSelf: "center",
                  bgcolor: theme.palette.secondary.dark,
                  mt: "auto",
                  width: "50%",
                }}
                type="submit"
              >
                {isLoading ? (
                  <Spinner height={"40px"} width="40px" />
                ) : (
                  "Sign up"
                )}
              </Button>
            </Stack>
          </form>
        </Stack>
      </Box>
    </Stack>
  );
};

SignUp.getLayout = (page: JSX.Element) => {
  return <Layout title="Signup">{page}</Layout>;
};

export default SignUp;
